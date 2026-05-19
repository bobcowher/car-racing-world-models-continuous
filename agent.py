import os
import subprocess
import gymnasium as gym
import torch
from buffer import ReplayBuffer
from utils import hard_update, soft_update, display_stacked_obs
from models.world_model import WorldModel
from models.actor import Actor
from models.critic import Critic
from torch.optim import Adam
import cv2
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import datetime
import random

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
        rewards = rewards.float()
        return states, actions, rewards, next_states, dones

    def _sample_imagined(self, batch_size, horizon):
        return self.agent.imagine_trajectory(batch_size, horizon)


class Agent:

    def __init__(self, env : gym.Env,
                       max_buffer_size : int = 10000,
                       world_model_batch_size = 8,
                       target_update_interval = 10000,
                       alpha : float = 0.1,
                       tau : float = 0.005) -> None:
        self.env = env
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.learning_rate = 0.0001
        self.alpha = alpha

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        obs, _ = self.env.reset()

        obs = self.process_observation(obs)

        assert self.env.action_space.shape is not None # Gets rid of pyright warnings
        self.n_actions = self.env.action_space.shape[0] # 3
        self.ac_hidden_size = 512

        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=obs.shape, input_device=self.device, output_device=self.device, action_dim=self.n_actions)

        self.world_model = WorldModel(observation_shape=obs.shape, embed_dim=1024, n_actions=self.n_actions).to(self.device)

        print(f"Observation shape: {obs.shape}")

        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.learning_rate)

        self.world_model_batch_size = world_model_batch_size

        self.critic = Critic(num_inputs=self.world_model.embed_dim, 
                             num_actions=self.n_actions, 
                             hidden_dim=self.ac_hidden_size, 
                             name=f"critic").to(device=self.device)

        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

        self.critic_target = Critic(num_inputs=self.world_model.embed_dim, 
                                    num_actions=self.n_actions, 
                                    hidden_dim=self.ac_hidden_size, 
                                    name=f"critic_target").to(self.device)
        
        hard_update(self.critic_target, self.critic)

        self.actor = Actor(num_inputs=self.world_model.embed_dim, 
                            num_actions=self.n_actions, 
                            hidden_dim=self.ac_hidden_size, 
                            action_space=self.env.action_space, 
                            name=f"policy").to(self.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)

        self.target_update_interval = target_update_interval

        self.gamma = 0.99
        self.tau = tau

        self.total_steps = 0
    
    def normalize_observation(self, obs):
        return obs / 255.0

    def process_observation(self, obs):
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
        obs = torch.from_numpy(obs).permute(2, 0, 1)
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

        # Lists to store rollout steps
        all_states      = []
        all_actions     = []
        all_rewards     = []
        all_next_states = []
        all_dones       = []

        current_embeds = embeds

        for _ in range(horizon):
            with torch.no_grad():
                action, _, _ = self.actor.sample(current_embeds)  # (batch_size, n_actions)

                next_embeds, rewards, dones = self.world_model.imagine_step(current_embeds, action)

                all_states.append(current_embeds)
                all_actions.append(action)
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

        for _ in range(epochs):
            obs, actions, rewards, next_obs, dones = self.memory.sample_buffer(batch_size)

            loss, loss_dict = self.world_model.compute_loss(obs, actions, rewards, next_obs, dones)

            self.world_model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.world_model_optimizer.step()

            total_loss += loss_dict["total"]
            total_recon += loss_dict["recon"]
            total_dynamics += loss_dict["dynamics"]
            total_reward += loss_dict["reward"]
            total_done += loss_dict["done"]

        return (
            total_loss / epochs,
            total_reward / epochs,
            total_done / epochs,
            total_recon / epochs,
            total_dynamics / epochs,
        )


    def train_actor_critic(self, sampler, horizon, batch_size, epochs):

        total_qf1_loss = 0.0
        total_qf2_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0 

        # Sample a batch from memory
        for _ in range(epochs):
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = sampler.sample(batch_size, horizon)

            state_batch = state_batch.float().to(self.device)
            next_state_batch = next_state_batch.float().to(self.device)
            action_batch = action_batch.float().to(self.device)
            reward_batch = reward_batch.float().to(self.device).unsqueeze(1)
            mask_batch = mask_batch.float().to(self.device).unsqueeze(1)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + (1.0 - mask_batch) * self.gamma * min_qf_next_target

            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            # Update the critic network
            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            # # Update the predictive model
            # self.predictive_model_optim.zero_grad()
            # prediction_error.backward()
            # self.predictive_model_optim.step()

            pi, log_pi, _ = self.actor.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()


            alpha_loss = torch.tensor(0.).to(self.device)

            soft_update(self.critic_target, self.critic, self.tau)
        
            total_qf1_loss += qf1_loss.item()
            total_qf2_loss += qf2_loss.item()
            total_actor_loss += actor_loss.item()
            total_alpha_loss += alpha_loss.item()

        return total_qf1_loss / epochs, total_qf2_loss / epochs, total_actor_loss / epochs, total_alpha_loss / epochs



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
            dummy_action = torch.zeros(num_samples, self.n_actions, device=self.device)
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
        self.actor.save_the_model("actor", verbose=True)
        self.critic.save_the_model("critic", verbose=True)

    def save_best(self, score, episode):
        path = "checkpoints/best.pt"
        torch.save({
            "episode": episode,
            "score": score,
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)
        print(f"Saved best checkpoint to {path} | episode: {episode} | score: {score:.1f}")

    def load(self):
        self.world_model.load_the_model("world_model", device=self.device)
        self.actor.load_the_model("actor", device=self.device)
        self.critic.load_the_model("critic", device=self.device)
        hard_update(self.critic_target, self.critic)

    def test(self, episodes=10):
        self.actor.eval()
        total_rewards = []

        for episode in range(episodes):
            obs, _ = self.env.reset()
            obs = self.process_observation(obs)
            done = False
            episode_reward = 0.0

            while not done:
                with torch.no_grad():
                    obs_t = obs.unsqueeze(0).float().to(self.device) / 255.0
                    embed = self.world_model.encode(obs_t).squeeze(1)
                    action = self.select_action(embed, evaluate=True)

                next_obs, reward, term, trunc, _ = self.env.step(action)
                next_obs = self.process_observation(next_obs)
                done = term or trunc
                episode_reward += float(reward)
                obs = next_obs

            total_rewards.append(episode_reward)
            print(f"Test episode {episode} | reward: {episode_reward:.1f}")

        avg = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg:.1f}")
        self.actor.train()
        return total_rewards

    def select_action(self, state, evaluate=False, min_gas=0.3):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.float().to(self.device)
        if state.dim() < 2:
            state = state.unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        action = action.detach().cpu().numpy()[0]
        action[1] = max(action[1], min_gas)  # gas is index 1, floor at min_gas
        return action

    def train(self, episodes=1, offline_training_epochs=1, batch_size=1, wm_batch_size=1, imagination_steps=None, real_ratio=0.5, warmup_episodes=5):

        rollout_steps = imagination_steps if imagination_steps is not None else batch_size

        try:
            run_tag = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            run_tag = 'unknown'
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}'

        writer = SummaryWriter(summary_writer_name)

        mixed_sampler = MixedSampler(self, real_ratio=real_ratio)
        best_score = float("-inf")

        for episode in range(episodes):
            obs, _ = self.env.reset()
            obs = self.process_observation(obs)

            done = False
            episode_reward = 0.0
            episode_loss = 0.0
            episode_steps = 0

            while not done:

                with torch.no_grad():
                    obs_t = obs.unsqueeze(0).float().to(self.device) / 255.0
                    embed = self.world_model.encode(obs_t).squeeze(1)  # (1, embed_dim)
                    action = self.select_action(embed)

                next_obs, reward, term, trunc, _ = self.env.step(action)

                next_obs = self.process_observation(next_obs)

                done = (term or trunc)

                self.memory.store_transition(obs, action, reward, next_obs, term)

                episode_reward += float(reward)
                episode_steps += 1

                obs = next_obs

            print(f"Episode {episode} | reward: {episode_reward:.1f} | steps: {episode_steps}")

            if episode_reward > best_score:
                best_score = episode_reward
                self.save_best(best_score, episode)

            # Adaptive real_ratio: start at 1.0 (pure real data), decay to floor by ep 400
            current_real_ratio = max(real_ratio, 1.0 - episode / 800.0)
            mixed_sampler.real_ratio = current_real_ratio

            current_ratio = [2, 2]

            total_combined_loss = 0.0
            total_reward_loss = 0.0
            total_done_loss = 0.0
            total_recon_loss = 0.0
            total_dynamics_loss = 0.0
            total_qf1_loss = 0.0
            total_qf2_loss = 0.0
            total_actor_loss = 0.0
            total_alpha_loss = 0.0
            wm_updates = 0
            ac_updates = 0

            for _ in range(offline_training_epochs):
                # World model updates
                for _ in range(current_ratio[0]):
                    combined_loss, reward_loss, done_loss, recon_loss, dynamics_loss = self.train_world_model(epochs=1, batch_size=wm_batch_size)
                    total_combined_loss += combined_loss
                    total_reward_loss += reward_loss
                    total_done_loss += done_loss
                    total_recon_loss += recon_loss
                    total_dynamics_loss += dynamics_loss
                    wm_updates += 1

                if episode >= warmup_episodes:
                    for _ in range(current_ratio[1]):
                        qf1_loss, qf2_loss, actor_loss, alpha_loss = self.train_actor_critic(mixed_sampler, rollout_steps, batch_size, epochs=1)
                        total_qf1_loss += qf1_loss
                        total_qf2_loss += qf2_loss
                        total_actor_loss += actor_loss
                        total_alpha_loss += alpha_loss
                        ac_updates += 1

            avg_combined_loss = total_combined_loss / wm_updates if wm_updates > 0 else 0.0
            avg_reward_loss = total_reward_loss / wm_updates if wm_updates > 0 else 0.0
            avg_done_loss = total_done_loss / wm_updates if wm_updates > 0 else 0.0
            avg_recon_loss = total_recon_loss / wm_updates if wm_updates > 0 else 0.0
            avg_dynamics_loss = total_dynamics_loss / wm_updates if wm_updates > 0 else 0.0
            episode_loss = (total_qf1_loss + total_qf2_loss) / (2 * ac_updates) if ac_updates > 0 else 0.0

            writer.add_scalar("World Model/combined_loss", avg_combined_loss, episode)
            writer.add_scalar("World Model/reconstruction_loss", avg_recon_loss, episode)
            writer.add_scalar("World Model/dynamics_loss", avg_dynamics_loss, episode)
            writer.add_scalar("World Model/reward_loss", avg_reward_loss, episode)
            writer.add_scalar("World Model/done_loss", avg_done_loss, episode)

            if ac_updates > 0:
                writer.add_scalar("SAC/qf1_loss", total_qf1_loss / ac_updates, episode)
                writer.add_scalar("SAC/qf2_loss", total_qf2_loss / ac_updates, episode)
                writer.add_scalar("SAC/actor_loss", total_actor_loss / ac_updates, episode)
                writer.add_scalar("SAC/alpha_loss", total_alpha_loss / ac_updates, episode)

            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/avg_critic_loss", episode_loss, episode)
            writer.add_scalar("Train/real_ratio", current_real_ratio, episode)
            writer.add_scalar("Train/best_score", best_score, episode)

            if episode % 10 == 0:
                self.evaluate_reconstruction(num_samples=4, filename="reconstruction_test.png")

            if episode % 10 == 0:
                self.save()


