import os
import gymnasium as gym
import torch
from buffer import ReplayBuffer
from utils import display_stacked_obs
from models.world_model import WorldModel
from models.q_model import QModel
import cv2
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import datetime
import random

def get_wm_q_ratio(episode):
    """Dynamic world model to Q-model training ratio based on episode.

    Keeps world model training strong throughout to track evolving data distribution.
    Never drops below 400 WM updates/episode to prevent WM degradation.
    """
    if episode < 25:
        return [4, 0]   # WM-only: build foundation
    elif episode < 100:
        return [3, 1]   # Start Q training
    elif episode < 250:
        return [2, 2]   # Balanced: let WM stabilize
    else:
        return [2, 3]   # Q-focused but WM stays strong (250-1200)


class MixedSampler:
    """Yields (states, actions, rewards, next_states, dones) in latent space.

    Each call randomly draws from the real replay buffer or world model imagination
    based on real_ratio. Both sources return the same tensor shapes and reward scale.
    """

    def __init__(self, agent, real_ratio=0.5):
        self.agent = agent
        self.real_ratio = real_ratio

    def sample(self, batch_size, horizon):
        if random.random() < self.real_ratio:
            return self._sample_real(batch_size, horizon)
        return self._sample_imagined(batch_size, horizon)

    def _sample_real(self, batch_size, horizon):
        agent = self.agent
        # Sample batch_size*horizon to match imagined output size
        obs, actions, rewards, next_obs, dones = agent.memory.sample_buffer(batch_size * horizon)
        with torch.no_grad():
            states      = agent.world_model.encode(agent.normalize_observation(obs)).squeeze(1)
            next_states = agent.world_model.encode(agent.normalize_observation(next_obs)).squeeze(1)
        rewards = rewards / 100.0  # match imagined reward scale (reward head trained with /100 targets)
        return states, actions, rewards, next_states, dones

    def _sample_imagined(self, batch_size, horizon):
        return self.agent.imagine_trajectory(batch_size, horizon)


class Agent:

    def __init__(self, env : gym.Env,
                       max_buffer_size : int = 10000,
                       world_model_batch_size = 8,
                       target_update_interval = 10000) -> None:
        self.env = env
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        obs, info = self.env.reset()

        obs = self.process_observation(obs)

        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=obs.shape, n_actions=self.env.action_space.n, input_device=self.device, output_device=self.device)

        # print(torch.squeeze(obs).shape)

        self.world_model = WorldModel(observation_shape=obs.shape, embed_dim=1024, n_actions=self.env.action_space.n, embed_norm='layernorm').to(self.device)

        print(f"Observation shape: {obs.shape}")

        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=0.0001)

        self.world_model_batch_size = world_model_batch_size

        self.q_model = QModel(action_dim=self.env.action_space.n, hidden_dim=256, embed_dim=self.world_model.embed_dim).to(self.device)
        self.target_q_model = QModel(action_dim=self.env.action_space.n, hidden_dim=256, embed_dim=self.world_model.embed_dim).to(self.device)

        self.q_model_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=0.0001)

        self.target_update_interval = target_update_interval

        self.gamma = 0.99

        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.98

        self.imagine_epsilon = 1
        self.imagine_min_epsilon = 0.1
        self.imagine_epsilon_decay = 0.99

        self.total_steps = 0
    
    def normalize_observation(self, obs):
        return obs / 255.0

    def process_observation(self, obs):
        # obs = torch.tensor(obs, dtype=torch.float32).permute(2,0,1)

        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
        # obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) # let's do grayscale    
        # obs = torch.from_numpy(obs).permute(2, 0, 1).to(self.device)


        obs = torch.from_numpy(obs)

        obs = obs.permute(2, 0, 1)
        
        return obs 


    def imagine_trajectory(self, batch_size, horizon):
        """
        Imagine parallel trajectories in latent space (no decoding).
        
        Args:
            batch_size: Number of parallel trajectories to sample.
            horizon: Number of steps to roll out for each trajectory.

        Returns flattened tensors of (batch_size * horizon, ...)
        """
        # Sample a batch of starting observations
        obs, _, _, _, _ = self.memory.sample_buffer(batch_size)
        obs = self.normalize_observation(obs)

        # Encode initial observations to latent space
        with torch.no_grad():
            embeds = self.world_model.encode(obs).squeeze(1)  # (batch_size, embed_dim)

        embed_dim = self.world_model.embed_dim
        
        # Lists to store rollout steps
        all_states      = []
        all_actions     = []
        all_rewards     = []
        all_next_states = []
        all_dones       = []

        current_embeds = embeds

        for _ in range(horizon):
            # Select actions for the entire batch (epsilon-greedy)
            with torch.no_grad():
                # Get Q-values for current batch
                q_vals = self.q_model(current_embeds) # (batch_size, n_actions)
                best_actions = q_vals.argmax(dim=1)   # (batch_size,)
                
                # Apply epsilon-greedy across the batch
                random_actions = torch.randint(0, self.env.action_space.n, (batch_size,), device=self.device)
                exploring_mask = (torch.rand(batch_size, device=self.device) < self.imagine_epsilon).long()
                action_idx = exploring_mask * random_actions + (1 - exploring_mask) * best_actions
                
                action_onehot = F.one_hot(action_idx, num_classes=self.env.action_space.n).float()

                # Imagine next step in parallel
                next_embeds, rewards, dones = self.world_model.imagine_step(current_embeds, action_onehot)

                all_states.append(current_embeds)
                all_actions.append(action_idx)
                all_rewards.append(rewards.squeeze(-1))
                all_next_states.append(next_embeds)
                all_dones.append((dones.squeeze(-1) > 0.5).float())

                current_embeds = next_embeds

        # Concatenate and flatten for the Q-learner
        states      = torch.cat(all_states, dim=0)      # (batch_size * horizon, embed_dim)
        actions     = torch.cat(all_actions, dim=0)     # (batch_size * horizon)
        rewards     = torch.cat(all_rewards, dim=0)     # (batch_size * horizon)
        next_states = torch.cat(all_next_states, dim=0) # (batch_size * horizon, embed_dim)
        dones       = torch.cat(all_dones, dim=0)       # (batch_size * horizon)

        return states, actions, rewards, next_states, dones

    def train_world_model(self, epochs, batch_size):
        """Train world model with reconstruction + dynamics + prediction losses."""

        total_loss = 0.0
        total_recon = 0.0
        total_dynamics = 0.0
        total_reward = 0.0
        total_done = 0.0
        total_l1 = 0.0
        total_ssim = 0.0
        total_edge = 0.0

        for _ in range(epochs):
            obs, actions, rewards, next_obs, dones = self.memory.sample_buffer(batch_size)

            # Compute all losses
            loss, loss_dict = self.world_model.compute_loss(obs, actions, rewards, next_obs, dones)

            # Optimize
            self.world_model_optimizer.zero_grad()
            loss.backward()
            self.world_model_optimizer.step()

            # Track losses
            total_loss += loss_dict["total"]
            total_recon += loss_dict["recon"]
            total_dynamics += loss_dict["dynamics"]
            total_reward += loss_dict["reward"]
            total_done += loss_dict["done"]
            total_l1 += loss_dict["l1"]
            total_ssim += loss_dict["ssim"]
            total_edge += loss_dict["edge"]

        # Average losses
        avg_total = total_loss / epochs
        avg_recon = total_recon / epochs
        avg_dynamics = total_dynamics / epochs
        avg_reward = total_reward / epochs
        avg_done = total_done / epochs
        avg_l1 = total_l1 / epochs
        avg_ssim = total_ssim / epochs
        avg_edge = total_edge / epochs

        # Return format: combined, reward, done, recon, dynamics, l1, ssim, edge
        return avg_total, avg_reward, avg_done, avg_recon, avg_dynamics, avg_l1, avg_ssim, avg_edge

    


    def train_q_model_on_imagination(self, horizon, batch_size, epochs=1):
        """Train Q-model on imagined trajectories (in latent space)."""

        total_loss = 0
        total_imag_reward = 0

        for epoch in range(epochs):

            # Imagine parallel trajectories in latent space
            embeddings, actions, rewards, next_embeddings, dones = self.imagine_trajectory(batch_size, horizon)

            total_imag_reward += rewards.mean().item()

            actions = actions.unsqueeze(1).long()
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1).float()

            # Q-learning in latent space
            q_values = self.q_model(embeddings)
            q_sa     = q_values.gather(1, actions)

            with torch.no_grad():
                next_actions = torch.argmax(
                    self.q_model(next_embeddings), dim=1, keepdim=True
                )

                next_q = self.target_q_model(next_embeddings).gather(1, next_actions)
                targets = rewards + (1 - dones) * self.gamma * next_q

            loss = F.mse_loss(q_sa, targets)

            self.q_model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=1.0)
            self.q_model_optimizer.step()

            if self.total_steps % self.target_update_interval == 0:
                self.target_q_model.load_state_dict(self.q_model.state_dict())

            total_loss += loss.item()

            self.total_steps += 1

        return total_loss / epochs, total_imag_reward / epochs



    def train_q_model_on_mixed(self, sampler, horizon, batch_size, epochs=1):
        """Train Q-model using MixedSampler (real buffer or imagination per call)."""

        total_loss = 0.0
        total_reward = 0.0

        for _ in range(epochs):
            states, actions, rewards, next_states, dones = sampler.sample(batch_size, horizon)

            total_reward += rewards.mean().item()

            actions = actions.unsqueeze(1).long()
            rewards = rewards.unsqueeze(1)
            dones   = dones.unsqueeze(1).float()

            q_sa = self.q_model(states).gather(1, actions)

            with torch.no_grad():
                next_actions = self.q_model(next_states).argmax(dim=1, keepdim=True)
                next_q       = self.target_q_model(next_states).gather(1, next_actions)
                targets      = rewards + (1 - dones) * self.gamma * next_q

            loss = F.mse_loss(q_sa, targets)
            self.q_model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=1.0)
            self.q_model_optimizer.step()

            if self.total_steps % self.target_update_interval == 0:
                self.target_q_model.load_state_dict(self.q_model.state_dict())

            total_loss += loss.item()
            self.total_steps += 1

        return total_loss / epochs, total_reward / epochs

    def evaluate_reconstruction(self, num_samples=4, filename="reconstruction_test.png"):
        """Evaluate reconstruction quality by comparing original vs reconstructed observations.

        Args:
            num_samples: Number of observations to reconstruct
            filename: Output image path
        """
        if not self.memory.can_sample(num_samples):
            return

        # Sample observations from replay buffer
        obs, _, _, _, _ = self.memory.sample_buffer(num_samples)
        obs_normalized = obs.float() / 255.0

        with torch.no_grad():
            # Get reconstructions from world model
            dummy_action = torch.zeros(num_samples, self.env.action_space.n, device=self.device)
            recon, _, _, _, _ = self.world_model.forward(obs_normalized, dummy_action)

        # Prepare visualization pairs
        viz_pairs = []
        for i in range(num_samples):
            viz_pairs.append((f"original_{i}", obs_normalized[i].cpu()))
            viz_pairs.append((f"recon_{i}", recon[i].cpu()))

        # Save comparison image
        display_stacked_obs(viz_pairs, filename, num_frames=1)
        print(f"Saved reconstruction comparison to {filename}")

    def save(self):
        self.world_model.save_the_model("world_model", verbose=True)
        self.q_model.save_the_model("q_model", verbose=True)

    def load(self):
        self.world_model.load_the_model("world_model", device=self.device)
        self.q_model.load_the_model("q_model", device=self.device)
        self.target_q_model.load_the_model("q_model", device=self.device)

    def test(self, episodes=10):
        self.q_model.eval()
        total_rewards = []

        for episode in range(episodes):
            obs, _ = self.env.reset()
            obs = self.process_observation(obs)
            done = False
            episode_reward = 0.0

            while not done:
                # Encode observation to latent space before Q-model
                with torch.no_grad():
                    obs_t = obs.unsqueeze(0).float().to(self.device) / 255.0
                    embed = self.world_model.encode(obs_t).squeeze(1)  # (1, embed_dim)
                    action = self.q_model(embed).argmax(dim=1).item()

                next_obs, reward, term, trunc, _ = self.env.step(action)
                next_obs = self.process_observation(next_obs)
                done = term or trunc
                episode_reward += reward
                obs = next_obs

            total_rewards.append(episode_reward)
            print(f"Test episode {episode} | reward: {episode_reward:.1f}")

        avg = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg:.1f}")
        self.q_model.train()
        return total_rewards

    def train(self, episodes=1, offline_training_epochs=1, batch_size=1, num_batches=1, wm_batch_size=1, imagination_steps=None, real_ratio=0.5):

        rollout_steps = imagination_steps if imagination_steps is not None else batch_size

        run_tag = f'world_model_ote{offline_training_epochs}_bs{batch_size}_wmbs{wm_batch_size}_rollout{rollout_steps}_buf{self.memory.mem_size}'
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}'

        writer = SummaryWriter(summary_writer_name)

        sampler = MixedSampler(self, real_ratio=real_ratio)

        for episode in range(episodes):
            
            obs, info = self.env.reset()

            obs = self.process_observation(obs)

            done = False
            episode_reward = 0.0
            episode_loss = 0.0
            episode_steps = 0

            while not done:

                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    # Encode observation to latent space before Q-model
                    with torch.no_grad():
                        obs_t = obs.unsqueeze(0).float().to(self.device) / 255.0
                        embed = self.world_model.encode(obs_t).squeeze(1)  # (1, embed_dim)
                        action = self.q_model(embed).argmax(dim=1).item()

                next_obs, reward, term, trunc, info = self.env.step(action)

                next_obs = self.process_observation(next_obs)

                done = (term or trunc)

                self.memory.store_transition(obs, action, reward, next_obs, done)

                episode_reward += reward
                episode_steps += 1

                obs = next_obs

            # Adjust epsilon.
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.imagine_epsilon = max(self.imagine_min_epsilon, self.imagine_epsilon * self.imagine_epsilon_decay)

            # Log stats for the current training iteration 
            print(f"Episode {episode} | reward: {episode_reward:.1f} | epsilon: {self.epsilon:.3f} | steps: {episode_steps}")

            # Interleaved training with dynamic wm_q_ratio
            current_ratio = get_wm_q_ratio(episode)

            total_combined_loss = 0
            total_reward_loss = 0
            total_done_loss = 0
            total_next_frame_loss = 0
            total_dynamics_loss = 0
            total_l1_loss = 0
            total_ssim_loss = 0
            total_edge_loss = 0
            total_q_loss = 0
            total_imag_reward = 0
            wm_updates = 0
            q_updates = 0

            for offline_epoch in range(offline_training_epochs):
                # World model updates
                for _ in range(current_ratio[0]):
                    combined_loss, reward_loss, done_loss, recon_loss, dynamics_loss, l1_loss, ssim_loss, edge_loss = self.train_world_model(epochs=1, batch_size=wm_batch_size)
                    total_combined_loss += combined_loss
                    total_reward_loss += reward_loss
                    total_done_loss += done_loss
                    total_next_frame_loss += recon_loss
                    total_dynamics_loss += dynamics_loss
                    total_l1_loss += l1_loss
                    total_ssim_loss += ssim_loss
                    total_edge_loss += edge_loss
                    wm_updates += 1

                # Q-model updates (ratio[1]=0 means no Q training)
                for _ in range(current_ratio[1]):
                    q_loss, imag_reward = self.train_q_model_on_mixed(sampler, rollout_steps, batch_size, epochs=1)
                    total_q_loss += q_loss
                    total_imag_reward += imag_reward
                    q_updates += 1

            # Average the losses
            avg_combined_loss = total_combined_loss / wm_updates if wm_updates > 0 else 0
            avg_reward_loss = total_reward_loss / wm_updates if wm_updates > 0 else 0
            avg_done_loss = total_done_loss / wm_updates if wm_updates > 0 else 0
            avg_next_frame_loss = total_next_frame_loss / wm_updates if wm_updates > 0 else 0
            avg_dynamics_loss = total_dynamics_loss / wm_updates if wm_updates > 0 else 0
            avg_l1_loss = total_l1_loss / wm_updates if wm_updates > 0 else 0
            avg_ssim_loss = total_ssim_loss / wm_updates if wm_updates > 0 else 0
            avg_edge_loss = total_edge_loss / wm_updates if wm_updates > 0 else 0
            episode_loss = total_q_loss / q_updates if q_updates > 0 else 0

            # Log all losses
            writer.add_scalar("World Model/combined_loss", avg_combined_loss, episode)
            writer.add_scalar("World Model/reward_loss", avg_reward_loss, episode)
            writer.add_scalar("World Model/done_loss", avg_done_loss, episode)
            writer.add_scalar("World Model/reconstruction_loss", avg_next_frame_loss, episode)
            writer.add_scalar("World Model/dynamics_loss", avg_dynamics_loss, episode)
            writer.add_scalar("Reconstruction/l1_loss", avg_l1_loss, episode)
            writer.add_scalar("Reconstruction/ssim_loss", avg_ssim_loss, episode)
            writer.add_scalar("Reconstruction/edge_loss", avg_edge_loss, episode)
            writer.add_scalar("Train/wm_updates_per_episode", wm_updates, episode)
            writer.add_scalar("Train/q_updates_per_episode", q_updates, episode)
            writer.add_scalar("Train/target_update_interval", self.target_update_interval, episode)
            writer.add_scalar("Train/updates_per_cycle_wm", current_ratio[0], episode)
            writer.add_scalar("Train/updates_per_cycle_q", current_ratio[1], episode)

            if q_updates > 0:
                avg_imag_reward = total_imag_reward / q_updates
                real_reward_per_step = episode_reward / episode_steps if episode_steps > 0 else 0.0
                writer.add_scalar("Imagination/mean_reward_per_step", avg_imag_reward, episode)
                writer.add_scalar("Imagination/real_reward_per_step", real_reward_per_step, episode)
                writer.add_scalar("Imagination/vs_real_reward_diff", avg_imag_reward - real_reward_per_step, episode)

            if episode % 100 == 0:
                print(f"Completed episode {episode} - Reward loss: {avg_reward_loss}")

            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/epsilon", self.epsilon, episode)
            writer.add_scalar("Train/imagine_epsilon", self.imagine_epsilon, episode)
            writer.add_scalar("Train/avg_q_loss", episode_loss, episode)
            writer.add_scalar("Train/real_ratio", real_ratio, episode)

            if episode % 10 == 0:
                self.evaluate_reconstruction(num_samples=4, filename="reconstruction_test.png")

            if episode % 10 == 0:
                self.save()




