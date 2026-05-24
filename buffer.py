import torch
import os
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size, input_shape,
                 input_device, output_device='cpu', action_dim=1):
        self.mem_size = max_size
        self.mem_ctr  = 0

        override = os.getenv("REPLAY_BUFFER_MEMORY")

        if override in ["cpu", "cuda:0", "cuda:1"]:
            print("Received replay buffer memory override.")
            self.input_device = override
        else:
            self.input_device  = input_device

        print(f"Replay buffer memory on: {self.input_device}")

        self.output_device = output_device

        self.state_memory      = torch.zeros(
            (max_size, *input_shape), dtype=torch.uint8, device=self.input_device
        )
        self.next_state_memory = torch.zeros(
            (max_size, *input_shape), dtype=torch.uint8, device=self.input_device
        )
        self.action_memory     = torch.zeros((max_size, action_dim), dtype=torch.float32,
                                             device=self.input_device)
        self.reward_memory     = torch.zeros(max_size, dtype=torch.float32,
                                             device=self.input_device)
        # terminal_memory: true only on environment termination (not truncation).
        # Used as the bootstrapping mask — truncation should still bootstrap V(s').
        self.terminal_memory   = torch.zeros(max_size, dtype=torch.bool,
                                             device=self.input_device)
        # episode_done_memory: true on any episode boundary (term OR trunc).
        # Used by sample_nstep to stop reward accumulation at episode resets.
        self.episode_done_memory = torch.zeros(max_size, dtype=torch.bool,
                                               device=self.input_device)

    def can_sample(self, batch_size: int) -> bool:
        return self.mem_ctr >= batch_size * 10

    def store_transition(self, state, action, reward, next_state, terminal, episode_done):
        """
        terminal    — true only on true environment termination (suppresses bootstrapping).
        episode_done — true on any episode boundary (term or trunc); stops n-step rollout.
        """
        idx = self.mem_ctr % self.mem_size

        self.state_memory[idx]       = torch.as_tensor(state, dtype=torch.uint8, device=self.input_device)
        self.next_state_memory[idx]  = torch.as_tensor(next_state, dtype=torch.uint8, device=self.input_device)
        self.action_memory[idx]      = torch.as_tensor(action, dtype=torch.float32, device=self.input_device)
        self.reward_memory[idx]      = float(reward)
        self.terminal_memory[idx]    = bool(terminal)
        self.episode_done_memory[idx] = bool(episode_done)

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch   = torch.randint(0, max_mem, (batch_size,),
                                device=self.input_device, dtype=torch.int64)

        states      = self.state_memory[batch].to(self.output_device, dtype=torch.float32)
        next_states = self.next_state_memory[batch].to(self.output_device, dtype=torch.float32)
        rewards     = self.reward_memory[batch].to(self.output_device)
        dones       = self.terminal_memory[batch].to(self.output_device)
        actions     = self.action_memory[batch].to(self.output_device)

        return states, actions, rewards, next_states, dones

    def sample_nstep(self, batch_size, n, gamma):
        """Sample n-step discounted returns with correct episode boundary handling.

        Uses absolute transition indices to guarantee sampled windows are
        chronologically contiguous — prevents crossing the circular buffer's
        write edge after the buffer fills, which would mix stale and fresh data.

        Reward accumulation stops at any episode boundary (term or trunc).
        Bootstrapping mask suppresses only true terminations (not truncations).
        """
        filled = min(self.mem_ctr, self.mem_size)
        # Absolute index range: oldest kept transition to newest safe start.
        # safe: start + n <= mem_ctr so all n slots are written.
        abs_min = self.mem_ctr - filled          # oldest slot still in buffer
        abs_max = self.mem_ctr - n               # latest safe start
        if abs_max <= abs_min:
            abs_max = abs_min + 1                # guard during early fill

        abs_starts = torch.randint(abs_min, abs_max, (batch_size,),
                                   dtype=torch.int64, device=self.input_device)
        start_idx = abs_starts % self.mem_size

        states  = self.state_memory[start_idx].to(self.output_device, dtype=torch.float32)
        actions = self.action_memory[start_idx].to(self.output_device)

        G          = torch.zeros(batch_size, dtype=torch.float32, device=self.output_device)
        active     = torch.ones(batch_size,  dtype=torch.float32, device=self.output_device)
        terminated = torch.zeros(batch_size, dtype=torch.float32, device=self.output_device)
        last_idx   = start_idx.clone()

        for k in range(n):
            idx     = (abs_starts + k).to(torch.int64) % self.mem_size
            r       = self.reward_memory[idx].to(self.output_device)
            ep_done = self.episode_done_memory[idx].float().to(self.output_device)
            term    = self.terminal_memory[idx].float().to(self.output_device)

            G = G + active * (gamma ** k) * r

            still_active = active.bool()
            last_idx[still_active] = idx[still_active]

            terminated = terminated + active * term

            # Stop accumulating at any episode boundary (term or trunc)
            active = active * (1.0 - ep_done)

        # Bootstrap mask: 1 only on true terminal, 0 on truncation (still bootstraps)
        done_composite    = (terminated > 0).float()
        final_next_states = self.next_state_memory[last_idx].to(self.output_device, dtype=torch.float32)

        return states, actions, G, final_next_states, done_composite

    def print_stats(self):
        filled = min(self.mem_ctr, self.mem_size)
        tensors = [self.state_memory, self.next_state_memory,
                   self.action_memory, self.reward_memory,
                   self.terminal_memory, self.episode_done_memory]
        used_bytes  = sum(t.element_size() * t.numel() * filled / self.mem_size for t in tensors)
        total_bytes = sum(t.element_size() * t.numel() for t in tensors)
        print(f"{filled} memories loaded | "
              f"used: {used_bytes / 1e9:.3f} GB / {total_bytes / 1e9:.3f} GB")


class EpisodeReplayBuffer(ReplayBuffer):
    """
    ReplayBuffer extended with episode boundary tracking for sequence sampling.

    All existing methods (sample_buffer, sample_nstep) work unchanged.
    sample_sequences() adds GRU/BPTT support — returns contiguous (B, T, ...)
    tensors from within single episodes, never crossing boundaries.

    No extra obs storage: sequences slice from the existing state_memory /
    next_state_memory tensors, so memory cost is identical to ReplayBuffer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Deque of (episode_start, episode_end) absolute indices for completed episodes.
        # episode_end is exclusive (one past the last stored step).
        self._episodes: deque[tuple[int, int]] = deque()
        self._current_episode_start: int = 0

    def store_transition(self, state, action, reward, next_state, terminal, episode_done):
        super().store_transition(state, action, reward, next_state, terminal, episode_done)

        if episode_done:
            self._episodes.append((self._current_episode_start, self.mem_ctr))
            self._current_episode_start = self.mem_ctr
            # Drop episodes whose start slot has been overwritten by the circular buffer.
            oldest_valid_index = self.mem_ctr - self.mem_size
            while self._episodes and self._episodes[0][0] < oldest_valid_index:
                self._episodes.popleft()

    def can_sample_sequences(self, batch_size: int, sequence_length: int, min_episodes: int = 10) -> bool:
        num_sequences = batch_size // sequence_length
        return sum(1 for episode_start, episode_end in self._episodes
                   if (episode_end - episode_start) >= sequence_length) >= max(min_episodes, num_sequences)

    def sample_sequences(self, batch_size: int, sequence_length: int) -> dict:
        """
        Sample contiguous sequences from within single episodes.

        batch_size must be evenly divisible by sequence_length. num_sequences is
        derived as batch_size // sequence_length so the total number of training
        samples matches what the rest of the training loop expects.

        Returns dict of (num_sequences, sequence_length, ...) tensors. The temporal
        dimension is preserved — each of the num_sequences rows is an independent
        contiguous sequence from a single episode. The GRU processes each row
        separately; hidden state never crosses sequence boundaries.

            obs      (num_sequences, sequence_length, C, H, W) uint8
            next_obs (num_sequences, sequence_length, C, H, W) uint8
            actions  (num_sequences, sequence_length, n_actions) float32
            rewards  (num_sequences, sequence_length) float32
            dones    (num_sequences, sequence_length) float32
        """
        if batch_size % sequence_length != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be evenly divisible by sequence_length ({sequence_length}). "
                f"num_sequences is derived as batch_size // sequence_length so that the total number of "
                f"training samples equals batch_size."
            )

        num_sequences = batch_size // sequence_length

        valid_episodes = [
            (episode_start, episode_end)
            for episode_start, episode_end in self._episodes
            if (episode_end - episode_start) >= sequence_length
        ]

        obs_out      = []
        next_obs_out = []
        actions_out  = []
        rewards_out  = []
        dones_out    = []

        for _ in range(num_sequences):
            episode_start, episode_end = valid_episodes[int(torch.randint(len(valid_episodes), (1,)).item())]
            episode_length = episode_end - episode_start
            window_offset  = int(torch.randint(0, episode_length - sequence_length + 1, (1,)).item())
            window_start   = episode_start + window_offset

            indices = (torch.arange(sequence_length, dtype=torch.int64) + window_start) % self.mem_size
            indices = indices.to(self.input_device)

            obs_out.append(self.state_memory[indices])
            next_obs_out.append(self.next_state_memory[indices])
            actions_out.append(self.action_memory[indices])
            rewards_out.append(self.reward_memory[indices])
            dones_out.append(self.terminal_memory[indices].float())

        output_device = self.output_device
        return {
            "obs":      torch.stack(obs_out).to(output_device),        # (num_sequences, sequence_length, C, H, W) — temporal order preserved; GRU processes each row as an independent sequence
            "next_obs": torch.stack(next_obs_out).to(output_device),   # (num_sequences, sequence_length, C, H, W)
            "actions":  torch.stack(actions_out).to(output_device),    # (num_sequences, sequence_length, n_actions)
            "rewards":  torch.stack(rewards_out).to(output_device),    # (num_sequences, sequence_length)
            "dones":    torch.stack(dones_out).to(output_device),      # (num_sequences, sequence_length)
        }
