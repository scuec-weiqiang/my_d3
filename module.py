import torch.nn as nn
import torch

class DynamicsModel(nn.Module):
    def __init__(self, state_dim=3, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)  # 输出下一状态（2）和奖励（1）
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)  # 输出 [Δs, reward]