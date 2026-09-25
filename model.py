import torch
import torch.nn as nn
import torch.nn.functional as F

class StrategicDQN(nn.Module):
    def __init__(self, grid_size, action_size):
        super(StrategicDQN, self).__init__()
        
        # Calculate the total number of cells in the grid
        self.input_dim = grid_size * grid_size
        self.action_size = action_size
        
        # Fully Connected Layers
        # Layer 1: Takes the flattened grid as input
        self.fc1 = nn.Linear(self.input_dim, 128)
        
        # Layer 2: Hidden layer for processing strategic relationships
        self.fc2 = nn.Linear(128, 128)
        
        # Layer 3: Hidden layer to refine the tactical features
        self.fc3 = nn.Linear(128, 64)
        
        # Output Layer: Outputs the Q-value for each possible action (Up, Down, Left, Right)
        self.fc4 = nn.Linear(64, self.action_size)

    def forward(self, x):
        # x comes in as a 2D grid (batch_size, grid_size, grid_size)
        # We need to flatten it to (batch_size, input_dim)
        x = x.view(x.size(0), -1)
        
        # Pass through layers with ReLU activation for non-linearity
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Output layer has no activation function because Q-values can be negative or positive
        return self.fc4(x)