import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model import StrategicDQN # Importing the brain we just built

class DQNAgent:
    def __init__(self, grid_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.grid_size = grid_size
        self.action_size = action_size
        
        # Tactical Memory: Stores past mission simulations (state, action, reward, next_state, done)
        self.memory = deque(maxlen=2000) 
        
        # Strategy Hyperparameters
        self.gamma = gamma              # Discount rate: How much the agent cares about future strategic value
        self.epsilon = epsilon          # Exploration rate: 1.0 means 100% random moves initially
        self.epsilon_decay = epsilon_decay # How fast the agent stops exploring and starts exploiting its knowledge
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        
        # Hardware configuration for Google Colab
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main Network (Decision Maker) and Target Network (Stable baseline for learning)
        self.model = StrategicDQN(grid_size, action_size).to(self.device)
        self.target_model = StrategicDQN(grid_size, action_size).to(self.device)
        self.update_target_network() # Sync the weights initially
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss() # Measuring the error in Q-value predictions

    def update_target_network(self):
        """Copies the weights from the main model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Logs the outcome of a maneuver into tactical memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-Greedy Policy: Decides whether to explore the map or execute a known strategy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Random scouting maneuver
        
        # Exploit known intelligence
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(np.argmax(q_values.cpu().numpy()[0])) # Execute the move with the highest Q-value

    def replay(self, batch_size):
        """The core learning phase. Reviews past maneuvers to improve future decisions."""
        if len(self.memory) < batch_size:
            return # Wait until we have enough intelligence gathered
        
        # Sample a random batch of past experiences
        minibatch = random.sample(self.memory, batch_size)
        
        # Convert memory batch into PyTorch tensors
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([m[1] for m in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([m[4] for m in minibatch])).to(self.device)
        
        # 1. Predict Q-values for current states
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        # 2. Predict maximum Q-values for next states using the stable Target Network
        next_q = self.target_model(next_states).max(1)[0]
        
        # 3. Apply the Bellman Equation (If the mission is done, target_q is just the reward)
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # 4. Calculate loss and perform backpropagation
        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 5. Slowly decay epsilon so the agent relies more on its training over time
        #if self.epsilon > self.epsilon_min:
           # self.epsilon *= self.epsilon_decay