from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


class TacticalGridEnv:
    """Tactical 2D Grid Reinforcement Learning Environment.

    Cell feature encoding:
        0.0  : Empty terrain
        0.5  : Agent position
        1.0  : Objective / Target
        -0.5 : Threat Zone (high risk)
        -1.0 : Hard Obstacle (impassable)
    """

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_NAMES = ["Move UP", "Move DOWN", "Move LEFT", "Move RIGHT"]
    MOVE_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(
        self,
        grid_size: int = 5,
        max_steps: int = 25,
        threats: Optional[List[Tuple[int, int]]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        target_pos: Optional[Tuple[int, int]] = None,
        start_pos: Optional[Tuple[int, int]] = None,
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Default tactical mission layout
        self.default_start = start_pos or (0, 0)
        self.default_target = target_pos or (grid_size - 1, grid_size - 1)
        self.default_threats = (
            threats if threats is not None else [(2, 2), (1, 3)]
        )
        self.default_obstacles = (
            obstacles if obstacles is not None else [(1, 1), (3, 2)]
        )

        self.agent_pos = list(self.default_start)
        self.target_pos = tuple(self.default_target)
        self.threats = [tuple(t) for t in self.default_threats]
        self.obstacles = [tuple(o) for o in self.default_obstacles]

        self.action_space = type("ActionSpace", (), {"n": 4})()
        self.step_count = 0
        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        self._fig = None
        self._ax = None

    def _build_grid_state(self) -> np.ndarray:
        """Construct the normalized numerical grid matrix."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Draw obstacles
        for r, c in self.obstacles:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                grid[r, c] = -1.0

        # Draw threats
        for r, c in self.threats:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                grid[r, c] = -0.5

        # Draw objective
        tr, tc = self.target_pos
        if 0 <= tr < self.grid_size and 0 <= tc < self.grid_size:
            grid[tr, tc] = 1.0

        # Draw agent
        ar, ac = self.agent_pos
        if 0 <= ar < self.grid_size and 0 <= ac < self.grid_size:
            grid[ar, ac] = 0.5

        return grid

    def reset(
        self,
        start_pos: Optional[Tuple[int, int]] = None,
        target_pos: Optional[Tuple[int, int]] = None,
        threats: Optional[List[Tuple[int, int]]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new mission."""
        if start_pos is not None:
            self.default_start = tuple(start_pos)
        if target_pos is not None:
            self.default_target = tuple(target_pos)
        if threats is not None:
            self.default_threats = [tuple(t) for t in threats]
        if obstacles is not None:
            self.default_obstacles = [tuple(o) for o in obstacles]

        self.agent_pos = list(self.default_start)
        self.target_pos = tuple(self.default_target)
        self.threats = [tuple(t) for t in self.default_threats]
        self.obstacles = [tuple(o) for o in self.default_obstacles]

        self.step_count = 0
        self.grid = self._build_grid_state()

        info = {
            "step": self.step_count,
            "agent_pos": list(self.agent_pos),
            "target_pos": list(self.target_pos),
            "threats": [list(t) for t in self.threats],
            "obstacles": [list(o) for o in self.obstacles],
            "event": "MISSION_INIT",
        }
        return self.grid.copy(), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes a single tactical move."""
        self.step_count += 1
        dr, dc = self.MOVE_DELTAS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        done = False
        truncated = False
        event = "MOVE_OK"

        # Check grid boundaries
        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            reward = -5.0
            event = "BOUNDARY_COLLISION"
        # Check obstacle collision
        elif (new_r, new_c) in self.obstacles:
            reward = -5.0
            event = "OBSTACLE_COLLISION"
        else:
            # Valid coordinate movement
            self.agent_pos = [new_r, new_c]
            curr_pos = (new_r, new_c)

            if curr_pos == self.target_pos:
                reward = 100.0
                done = True
                event = "OBJECTIVE_SECURED"
            elif curr_pos in self.threats:
                reward = -10.0
                event = "THREAT_ZONE_ENTERED"
            else:
                reward = -1.0  # Step living cost for optimal path

        if not done and self.step_count >= self.max_steps:
            truncated = True
            event = "TIMEOUT"

        self.grid = self._build_grid_state()

        info = {
            "step": self.step_count,
            "agent_pos": list(self.agent_pos),
            "target_pos": list(self.target_pos),
            "threats": [list(t) for t in self.threats],
            "obstacles": [list(o) for o in self.obstacles],
            "event": event,
            "action_name": self.ACTION_NAMES[action],
        }
        return self.grid.copy(), float(reward), done, truncated, info

    def render(self, mode: str = "console") -> None:
        """Render the environment state either in console or matplotlib."""
        if mode == "console":
            symbols = {
                0.0: ". ",
                0.5: "A ",
                1.0: "T ",
                -0.5: "! ",
                -1.0: "# ",
            }
            border = "+-" * self.grid_size + "+"
            print(border)
            for r in range(self.grid_size):
                row_str = "|"
                for c in range(self.grid_size):
                    val = self.grid[r, c]
                    row_str += symbols.get(val, "? ")
                print(row_str + "|")
            print(border)
            return

        plt.ion()
        if not plt.fignum_exists(1):
            self._fig, self._ax = plt.subplots(figsize=(5, 5), num=1)
        plt.clf()
        plt.matshow(self.grid, fignum=1, cmap="viridis")
        plt.xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        plt.yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        plt.grid(which="minor", color="black", linestyle="-", linewidth=2)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Tactical Grid Mission (Step {self.step_count})")
        plt.draw()
        plt.pause(0.1)
