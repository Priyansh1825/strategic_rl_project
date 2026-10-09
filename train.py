import sys
from typing import Dict, List

import numpy as np

from agent import DQNAgent
from tactical_env import TacticalGridEnv


def train_tactical_agent(
    episodes: int = 400,
    batch_size: int = 32,
    target_update_freq: int = 10,
    weights_path: str = "strategic_agent_weights.pth",
) -> Dict[str, List[float]]:
    """Trains DQN agent on TacticalGridEnv and saves checkpoint."""
    env = TacticalGridEnv(grid_size=5, max_steps=30)
    agent = DQNAgent(
        grid_size=5,
        action_size=env.action_space.n,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.985,
        epsilon_min=0.02,
        memory_size=10000,
    )

    history: Dict[str, List[float]] = {
        "episode_rewards": [],
        "losses": [],
        "success_rates": [],
    }

    recent_successes = []

    print(
        f"[*] Training Tactical DQN on {agent.device} for {episodes} eps..."
    )

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        truncated = False
        ep_losses = []

        while not done and not truncated:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss = agent.replay(batch_size)
            if loss is not None:
                ep_losses.append(loss)

        if ep % target_update_freq == 0:
            agent.update_target_network()

        is_success = 1.0 if total_reward > 50.0 else 0.0
        recent_successes.append(is_success)
        if len(recent_successes) > 50:
            recent_successes.pop(0)

        rolling_success = float(np.mean(recent_successes)) * 100.0
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        history["episode_rewards"].append(total_reward)
        history["losses"].append(avg_loss)
        history["success_rates"].append(rolling_success)

        if ep % 25 == 0 or ep == episodes:
            print(
                f"Episode {ep:03d}/{episodes:03d} | "
                f"Reward: {total_reward:6.1f} | "
                f"Loss: {avg_loss:6.3f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Win Rate (L50): {rolling_success:5.1f}%"
            )

    agent.save(weights_path)
    print(f"[+] Training completed. Weights saved to: {weights_path}")
    return history


if __name__ == "__main__":
    episodes_count = 350
    if len(sys.argv) > 1:
        try:
            episodes_count = int(sys.argv[1])
        except ValueError:
            pass

    train_tactical_agent(episodes=episodes_count)
