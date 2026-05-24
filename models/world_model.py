import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder, Decoder
from models.base import BaseModel
from models.dynamics_model import DynamicsModel
from models.ssim_loss import ssim_loss


def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class WorldModel(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=1024, gru_dim=512, n_actions=4, feature_dim=None):
        super().__init__()

        if feature_dim is None:
            feature_dim = embed_dim

        self.encoder = Encoder(observation_shape=observation_shape, embed_dim=embed_dim)
        self.decoder = Decoder(observation_shape=observation_shape, embed_dim=feature_dim,
                               conv_output_shape=self.encoder.get_output_shape(),
                               conv_channels=self.encoder.get_conv_channels())

        self.dynamics = DynamicsModel(embed_dim=embed_dim, n_actions=n_actions, hidden_dim=2048)

        self.embed_norm_layer = nn.LayerNorm(embed_dim)

        self.reward_pred = nn.Linear(gru_dim + n_actions, 1)
        self.done_pred = nn.Linear(gru_dim + n_actions, 1)

        self.embed_dim = embed_dim
        self.gru_dim = gru_dim
        self.n_actions = n_actions

        self.gru = nn.GRU(input_size=embed_dim, hidden_size=gru_dim, batch_first=True)

        print(f"World Model initialized. Input shape: {observation_shape}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Dynamics: embed + action → next_embed")
        print(f"  Prediction heads: reward, done")


    def normalize_embedding(self, embed):
        return self.embed_norm_layer(embed)

    def encode(self, obs, hidden=None):
        # If obs is [B, C, H, W], add sequence dimension -> [B, 1, C, H, W]
        if obs.ndim == 4:
            obs = obs.unsqueeze(1)

        batch_size, sequence_length = obs.shape[:2]
        obs_flat = obs.view(batch_size * sequence_length, *obs.shape[2:])
        embed_flat = self.encoder(obs_flat)

        # Normalize embeddings
        embed_flat = self.normalize_embedding(embed_flat)

        embeds = embed_flat.view(batch_size, sequence_length, -1)

        gru_out, hidden = self.gru(embeds, hidden) 
        h_t = gru_out.squeeze(1)

        return embeds, h_t, hidden

    def decode(self, embeds):
        return self.decoder(embeds)

    def imagine_step(self, embed, h_t, action_onehot):
        """
        Imagination step in latent space (no decoding).

        Args:
            embed: (B, embed_dim) current state embedding
            action_onehot: (B, n_actions) one-hot encoded action

        Returns:
            next_embed: (B, embed_dim) predicted next state embedding
            reward: (B, 1) predicted reward
            done: (B, 1) predicted done probability
        """
        # Predict next embedding and normalize it
        next_embed = self.dynamics(embed, action_onehot)
        next_embed = self.normalize_embedding(next_embed)

        gru_out, next_hidden_state = self.gru(next_embed.unsqueeze(1), h_t.unsqueeze(0))
        next_h_t = gru_out.squeeze(1)

        # Predict reward and done
        embed_action = torch.cat([h_t, action_onehot], dim=-1)
        reward = self.reward_pred(embed_action)
        done = torch.sigmoid(self.done_pred(embed_action))

        return next_embed, next_h_t, next_hidden_state, reward, done

    def compute_loss_sequential(self, batch: dict):
        """
        Compute world model losses over sequences of contiguous transitions.

        Args:
            batch: dict from EpisodeReplayBuffer.sample_sequences() with keys:
                obs      (N, T, C, H, W) uint8
                actions  (N, T, n_actions) float32
                rewards  (N, T) float32
                dones    (N, T) float32

        The GRU hidden state is initialized to zero at the start of each sequence
        and flows forward across all T steps. No BPTT across sequence boundaries.

        Returns:
            combined_loss: scalar
            loss_dict: dict of individual loss values
        """
        obs = batch["obs"].float() / 255.0   # (N, T, C, H, W)
        actions = batch["actions"]            # (N, T, n_actions)
        rewards = batch["rewards"]            # (N, T)
        dones = batch["dones"]               # (N, T)

        num_sequences, sequence_length = obs.shape[:2]

        # Encode full sequences. hidden=None zeros the GRU for each sequence.
        # embeds:      (N, T, embed_dim)
        # gru_outputs: (N, T, gru_dim) — hidden state at every timestep
        embeds, gru_outputs, _ = self.encode(obs)

        # === Reconstruction loss: decode every frame ===
        embeds_flat = embeds.reshape(num_sequences * sequence_length, -1)
        obs_flat    = obs.reshape(num_sequences * sequence_length, *obs.shape[2:])
        recon       = self.decode(embeds_flat)
        recon_loss  = (F.l1_loss(recon, obs_flat)
                       + 0.2 * ssim_loss(recon, obs_flat)
                       + 0.1 * gradient_loss(recon, obs_flat))

        # === Dynamics loss: embeds[t] + actions[t] → predict embeds[t+1] ===
        # Covers t=0..T-2 (T-1 prediction pairs per sequence).
        num_pairs         = num_sequences * (sequence_length - 1)
        embeds_current    = embeds[:, :-1, :].reshape(num_pairs, -1)
        actions_current   = actions[:, :-1, :].reshape(num_pairs, -1)
        next_embed_pred   = self.normalize_embedding(self.dynamics(embeds_current, actions_current))
        next_embed_target = embeds[:, 1:, :].reshape(num_pairs, -1).detach()
        dynamics_loss     = F.mse_loss(next_embed_pred, next_embed_target)

        # === Reward and done prediction from (gru_outputs[t], actions[t]) ===
        gru_outputs_flat = gru_outputs.reshape(num_sequences * sequence_length, -1)
        actions_flat     = actions.reshape(num_sequences * sequence_length, -1)
        embed_action     = torch.cat([gru_outputs_flat, actions_flat], dim=-1)

        reward_pred  = self.reward_pred(embed_action)
        reward_loss  = F.mse_loss(reward_pred.squeeze(-1), rewards.reshape(num_sequences * sequence_length))

        done_pred    = torch.sigmoid(self.done_pred(embed_action))
        done_loss    = F.binary_cross_entropy(done_pred.squeeze(-1), dones.reshape(num_sequences * sequence_length))

        combined_loss = (1.0 * recon_loss
                         + 1.0 * dynamics_loss
                         + 2.0 * reward_loss
                         + 0.5 * done_loss)

        return combined_loss, {
            "total":    combined_loss.item(),
            "recon":    recon_loss.item(),
            "dynamics": dynamics_loss.item(),
            "reward":   reward_loss.item(),
            "done":     done_loss.item(),
        }
    



