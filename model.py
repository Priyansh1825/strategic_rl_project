import torch.nn as nn
import torch.nn.functional as F


class StrategicDQN(nn.Module):
    """Deep Q-Network for tactical spatial grid decision making."""

    def __init__(self, grid_size: int = 5, action_size: int = 4):
        super(StrategicDQN, self).__init__()
        self.grid_size = grid_size
        self.input_dim = grid_size * grid_size
        self.action_size = action_size

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.action_size)

    def forward(self, x):
        """Forward pass expecting (batch, grid, grid) or flat."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        elif x.dim() == 1:
            x = x.unsqueeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
