# import torch.nn as nn
# import torch.nn.functional as F
# from models.base import BaseModel
#
# class QModel(BaseModel):
#     def __init__(self, action_dim, hidden_dim=256, embed_dim=1024):
#         """
#         Q-model that operates on latent embeddings instead of raw images.
#
#         Args:
#             action_dim: Number of possible actions
#             hidden_dim: Hidden layer dimension
#             embed_dim: Dimension of world model embeddings (default 1024)
#         """
#         super(QModel, self).__init__()
#
#         self.embed_dim = embed_dim
#
#         # MLP layers for latent embeddings
#         self.fc1 = nn.Linear(embed_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.output = nn.Linear(hidden_dim, action_dim)
#
#         # Initialize weights
#         self.apply(self.weights_init)
#
#         print(f"Q-Model initialized (latent-based):")
#         print(f"  Input: {embed_dim}-dim embeddings")
#         print(f"  Hidden: {hidden_dim}")
#         print(f"  Output: {action_dim} actions")
#
#     def forward(self, embeddings):
#         """
#         Forward pass through Q-network.
#
#         Args:
#             embeddings: (B, embed_dim) latent embeddings from world model
#
#         Returns:
#             q_values: (B, action_dim) Q-values for each action
#         """
#         x = F.relu(self.fc1(embeddings))
#         x = F.relu(self.fc2(x))
#         q_values = self.output(x)
#
#         return q_values
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_normal_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
