from numpy import int32, rec
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

    def __init__(self, observation_shape=(), embed_dim=1024, action_dim=128, n_actions=4, feature_dim=None):
        super().__init__()

        if feature_dim is None:
            feature_dim = embed_dim

        self.encoder = Encoder(observation_shape=observation_shape, embed_dim=embed_dim)
        self.decoder = Decoder(observation_shape=observation_shape, embed_dim=feature_dim,
                               conv_output_shape=self.encoder.get_output_shape(),
                               conv_channels=self.encoder.get_conv_channels())

        self.dynamics = DynamicsModel(embed_dim=embed_dim, n_actions=n_actions, hidden_dim=2048)

        self.embed_norm_layer = nn.LayerNorm(embed_dim)

        self.reward_pred = nn.Linear(embed_dim + n_actions, 1)
        self.done_pred = nn.Linear(embed_dim + n_actions, 1)

        self.embed_dim = embed_dim
        self.n_actions = n_actions

        print(f"World Model initialized. Input shape: {observation_shape}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Dynamics: embed + action → next_embed")
        print(f"  Prediction heads: reward, done")


    def normalize_embedding(self, embed):
        return self.embed_norm_layer(embed)

    def encode(self, obs):
        # If obs is [B, C, H, W], add sequence dimension -> [B, 1, C, H, W]
        if obs.ndim == 4:
            obs = obs.unsqueeze(1)

        batch_size, sequence_length = obs.shape[:2]
        obs_flat = obs.view(batch_size * sequence_length, *obs.shape[2:])
        embed_flat = self.encoder(obs_flat)

        # Normalize embeddings
        embed_flat = self.normalize_embedding(embed_flat)

        embeds = embed_flat.view(batch_size, sequence_length, -1)

        return embeds

    def decode(self, embeds):
        return self.decoder(embeds)

    def imagine_step(self, embed, action_onehot):
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

        # Predict reward and done
        embed_action = torch.cat([embed, action_onehot], dim=-1)
        reward = self.reward_pred(embed_action)
        done = torch.sigmoid(self.done_pred(embed_action))

        return next_embed, reward, done

    def compute_loss(self, obs, actions, rewards, next_obs, dones):
        """
        Compute all world model losses.

        Args:
            obs: (B, C, H, W) uint8 observations
            actions: (B,) action indices
            rewards: (B,) rewards
            next_obs: (B, C, H, W) uint8 next observations
            dones: (B,) done flags

        Returns:
            combined_loss: scalar total loss
            loss_dict: dictionary of individual losses
        """
        # Normalize observations
        obs_normalized = obs.float() / 255.0
        next_obs_normalized = next_obs.float() / 255.0

        if obs_normalized.ndim == 5:
            obs_normalized = obs_normalized.squeeze(1)
        if next_obs_normalized.ndim == 5:
            next_obs_normalized = next_obs_normalized.squeeze(1)

        # Convert actions to one-hot
        batch_size = obs.shape[0]
        action_onehot = F.one_hot(actions.long(), num_classes=self.n_actions).float()

        # Forward pass
        recon, embeds, next_embed_pred, reward_pred, done_pred = self.forward(obs_normalized, action_onehot)

        # === 1. Reconstruction Loss ===
        recon_loss = F.l1_loss(recon, obs_normalized) + 0.2 * ssim_loss(recon, obs_normalized) + 0.1 * gradient_loss(recon, obs_normalized)

        # === 2. Dynamics Loss ===
        # Encode next observation to get target embedding
        next_embeds = self.encode(next_obs_normalized)  # (B, 1, embed_dim)
        next_embed_target = next_embeds.view(-1, next_embeds.shape[-1])  # (B, embed_dim)

        # MSE between predicted and actual next embedding
        dynamics_loss = F.mse_loss(next_embed_pred, next_embed_target.detach())

        # === 3. Reward Loss ===
        reward_loss = F.mse_loss(reward_pred.squeeze(-1), rewards.float())

        # === 4. Done Loss ===
        # Binary classification
        done_loss = F.binary_cross_entropy(done_pred.squeeze(-1), dones.float())

        # === Combined Loss ===
        combined_loss = (
            1.0 * recon_loss +
            1.0 * dynamics_loss +
            2.0 * reward_loss +
            0.5 * done_loss
        )

        return combined_loss, {
            "total": combined_loss.item(),
            "recon": recon_loss.item(),
            "dynamics": dynamics_loss.item(),
            "reward": reward_loss.item(),
            "done": done_loss.item(),
        }


    def forward(self, obs, action_onehot):
        """
        Full forward pass through world model.

        Args:
            obs: (B, C, H, W) observations (uint8 or normalized float)
            action_onehot: (B, n_actions) one-hot encoded actions

        Returns:
            recon: (B, C, H, W) reconstructed observation
            embeds: (B, 1, embed_dim) current state embeddings
            next_embed_pred: (B, embed_dim) predicted next state embedding
            reward_pred: (B, 1) predicted reward
            done_pred: (B, 1) predicted done probability
        """
        # Encode observation to latent state
        embeds = self.encode(obs)  # (B, 1, embed_dim)

        # Decode for reconstruction
        embeds_flat = embeds.view(-1, embeds.shape[-1])  # (B, embed_dim)
        recon = self.decode(embeds_flat)  # (B, C, H, W)

        # Flatten embeddings for predictions
        embed = embeds_flat  # (B, embed_dim)

        # Predict next embedding using dynamics model and normalize it
        next_embed_pred = self.dynamics(embed, action_onehot)  # (B, embed_dim)
        next_embed_pred = self.normalize_embedding(next_embed_pred)

        # Predict reward from current state + action
        embed_action = torch.cat([embed, action_onehot], dim=-1)
        reward_pred = self.reward_pred(embed_action)

        # Predict done from current state + action
        done_pred = torch.sigmoid(self.done_pred(embed_action))  # (B, 1) in [0, 1]

        return recon, embeds, next_embed_pred, reward_pred, done_pred
    



