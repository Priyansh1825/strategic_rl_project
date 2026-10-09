from collections import deque
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import StrategicDQN


class DQNAgent:
    """Deep Q-Network Agent for tactical spatial pathfinding."""

    def __init__(
        self,
        grid_size: int = 5,
        action_size: int = 4,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.02,
        memory_size: int = 5000,
    ):
        self.grid_size = grid_size
        self.action_size = action_size

        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = StrategicDQN(grid_size, action_size).to(self.device)
        self.target_model = StrategicDQN(grid_size, action_size).to(
            self.device
        )
        self.update_target_network()

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = nn.MSELoss()

    def update_target_network(self) -> None:
        """Syncs target network weights with primary model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stores transition in tactical replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-greedy decision policy."""
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = (
            torch.FloatTensor(state).unsqueeze(0).to(self.device)
        )
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        return int(torch.argmax(q_values, dim=1).item())

    def get_q_values(self, state: np.ndarray) -> List[float]:
        """Calculates Q-values across all discrete actions."""
        state_tensor = (
            torch.FloatTensor(state).unsqueeze(0).to(self.device)
        )
        self.model.eval()
        with torch.no_grad():
            q_vals = self.model(state_tensor).cpu().numpy()[0]
        self.model.train()
        return [float(v) for v in q_vals]

    def replay(self, batch_size: int = 32) -> Optional[float]:
        """Experience replay batch learning step."""
        if len(self.memory) < batch_size:
            return None

        minibatch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor(
            np.array([m[0] for m in minibatch])
        ).to(self.device)
        actions = torch.LongTensor(
            np.array([m[1] for m in minibatch])
        ).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(
            np.array([m[2] for m in minibatch])
        ).to(self.device)
        next_states = torch.FloatTensor(
            np.array([m[3] for m in minibatch])
        ).to(self.device)
        dones = torch.FloatTensor(
            np.array([m[4] for m in minibatch])
        ).to(self.device)

        # Current Q-values for selected actions
        current_q = self.model(states).gather(1, actions).squeeze(1)

        # Max Q-values from target network for next states
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (1.0 - dones))

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Anneal exploration rate
        if self.epsilon > self.epsilon_min:
            new_eps = self.epsilon * self.epsilon_decay
            self.epsilon = max(self.epsilon_min, new_eps)

        return float(loss.item())

    def save(self, filepath: str) -> None:
        """Saves policy model state dict."""
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """Loads weights with safe device mapping."""
        self.model.load_state_dict(
            torch.load(filepath, map_location=self.device, weights_only=True)
        )
        self.update_target_network()
        self.model.eval()
