import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models.base import *
import os

class Critic(BaseModel):
    def __init__(self, num_inputs, num_actions, hidden_dim, checkpoint_dir='checkpoints', name='q_network'):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.output1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)
        self.output2 = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.ln1(self.linear1(xu)))
        x1 = F.relu(self.ln2(self.linear2(x1)))
        x1 = self.output1(x1)

        x2 = F.relu(self.ln4(self.linear4(xu)))
        x2 = F.relu(self.ln5(self.linear5(x2)))
        x2 = self.output2(x2)

        return x1, x2

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
