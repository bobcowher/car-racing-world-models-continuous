import torch
import os

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

        # States (uint8 saves ~4× RAM vs float32)
        self.state_memory      = torch.zeros(
            (max_size, *input_shape), dtype=torch.uint8, device=self.input_device
        )
        self.next_state_memory      = torch.zeros(
            (max_size, *input_shape), dtype=torch.uint8, device=self.input_device
        )

        self.action_memory  = torch.zeros((max_size, action_dim), dtype=torch.float32,
                                          device=self.input_device)
        self.reward_memory  = torch.zeros(max_size, dtype=torch.float32,
                                          device=self.input_device)
        # terminal_memory: true only on environment termination (not time-limit truncation).
        # Used as the bootstrapping mask in Q-targets — truncation should still bootstrap.
        self.terminal_memory = torch.zeros(max_size, dtype=torch.bool,
                                           device=self.input_device)
        # episode_done_memory: true on any episode boundary (term OR trunc).
        # Used by sample_nstep() to stop rolling forward across episode resets.
        self.episode_done_memory = torch.zeros(max_size, dtype=torch.bool,
                                               device=self.input_device)

    def can_sample(self, batch_size: int) -> bool:
        """Require at least 10×batch_size transitions before sampling."""
        return self.mem_ctr >= batch_size * 10

    def store_transition(self, state, action, reward, next_state, terminal, episode_done):
        """Write a transition in-place on `input_device`.

        terminal    — true only on environment termination (suppresses bootstrapping).
        episode_done — true on any episode boundary (term or trunc); stops n-step rollout.
        """
        idx = self.mem_ctr % self.mem_size

        self.state_memory[idx]      = torch.as_tensor(
            state, dtype=torch.uint8, device=self.input_device)
        self.next_state_memory[idx] = torch.as_tensor(
            next_state, dtype=torch.uint8, device=self.input_device)

        self.action_memory[idx]      = torch.as_tensor(action, dtype=torch.float32, device=self.input_device)
        self.reward_memory[idx]      = float(reward)
        self.terminal_memory[idx]    = bool(terminal)
        self.episode_done_memory[idx] = bool(episode_done)

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        """Return tensors ready for training (on `output_device`)."""
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
        """Sample batch with n-step discounted returns.

        Rolls forward n steps from each sampled start index, accumulating
        γ^k * r_{t+k}. Stops accumulating at any episode boundary (term or trunc)
        so returns never cross episode resets.

        done_composite is 1 only when a true terminal was encountered — truncation
        boundaries still allow Q-value bootstrapping.
        """
        max_mem = min(self.mem_ctr, self.mem_size)
        # Guard: exclude the last n slots so the rollout window never reaches
        # unwritten positions (early fill) or crosses the circular-buffer write edge.
        safe_max = max(1, max_mem - n)
        starts = torch.randint(0, safe_max, (batch_size,), device=self.input_device)

        states  = self.state_memory[starts].to(self.output_device, dtype=torch.float32)
        actions = self.action_memory[starts].to(self.output_device)

        G          = torch.zeros(batch_size, dtype=torch.float32, device=self.output_device)
        active     = torch.ones(batch_size,  dtype=torch.float32, device=self.output_device)
        terminated = torch.zeros(batch_size, dtype=torch.float32, device=self.output_device)
        last_idx   = starts.clone()

        for k in range(n):
            idx      = (starts + k) % self.mem_size
            r        = self.reward_memory[idx].to(self.output_device)
            ep_done  = self.episode_done_memory[idx].float().to(self.output_device)
            term     = self.terminal_memory[idx].float().to(self.output_device)

            G = G + active * (gamma ** k) * r

            still_active = active.bool()
            last_idx[still_active] = idx[still_active]

            # Track true terminals encountered while still in the rollout window
            terminated = terminated + active * term

            # Stop rolling forward at any episode boundary (term or trunc)
            active = active * (1.0 - ep_done)

        # Bootstrapping mask: suppress only on true termination, not time-limit truncation
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
