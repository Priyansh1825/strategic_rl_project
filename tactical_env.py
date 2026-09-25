import numpy as np
import matplotlib.pyplot as plt

class TacticalGridEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.action_space = type('ActionSpace', (), {'n': 4})()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        state = self.grid.copy()
        info = {}
        return state, info

    def step(self, action):
        next_state = self.grid.copy()
        reward = 0.0
        done = False
        truncated = False
        info = {}
        return next_state, reward, done, truncated, info

    def render(self):
        plt.ion()
        if not plt.fignum_exists(1):
            plt.figure(1, figsize=(5, 5))
        plt.clf()
        plt.matshow(self.grid, fignum=1, cmap='viridis')
        plt.xticks(np.arange(-.5, self.grid_size, 1), minor=True)
        plt.yticks(np.arange(-.5, self.grid_size, 1), minor=True)
        plt.grid(which='minor', color='black', linestyle='-', linewidth=2)
        plt.xticks([])
        plt.yticks([])
        plt.title("Tactical AI Demo")
        plt.draw()
        plt.pause(0.5)