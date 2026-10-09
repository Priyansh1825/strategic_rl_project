import os
import time

import matplotlib.pyplot as plt
import numpy as np

from agent import DQNAgent
from tactical_env import TacticalGridEnv


def run_tactical_demo(
    weights_path: str = "strategic_agent_weights.pth",
    render_mode: str = "console",
    delay_sec: float = 0.5,
    max_steps: int = 25,
) -> bool:
    """Executes a simulated tactical mission using the trained policy."""
    grid_size = 5
    env = TacticalGridEnv(grid_size=grid_size, max_steps=max_steps)
    agent = DQNAgent(grid_size=grid_size, action_size=env.action_space.n)

    # 1. Safely load pre-trained weights if available
    if os.path.exists(weights_path):
        agent.load(weights_path)
        print(f"[+] Loaded tactical weights from: {weights_path}")
    else:
        print(f"[!] Warning: '{weights_path}' not found. Using untrained.")

    state, info = env.reset()
    done = False
    truncated = False
    step = 0
    total_reward = 0.0
    last_reward: float = 0.0

    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 55)
    print("       --- TACTICAL MISSION DEPLOYMENT ---")
    print("=" * 55)
    time.sleep(1.0)

    # 2. Live Simulation Loop
    while not done and not truncated and step < max_steps:
        os.system("cls" if os.name == "nt" else "clear")

        print("=" * 55)
        print(
            f" STEP {step + 1}/{max_steps} | Agent: {env.agent_pos} "
            f"-> Target: {env.target_pos}"
        )
        print("=" * 55)

        if render_mode == "console":
            env.render(mode="console")
        elif render_mode == "gui":
            env.render(mode="gui")

        # Query Q-values from policy network
        q_values = agent.get_q_values(state)
        action = int(np.argmax(q_values))
        action_name = TacticalGridEnv.ACTION_NAMES[action]

        print("\n--- TACTICAL RADAR ---")
        for i, name in enumerate(TacticalGridEnv.ACTION_NAMES):
            marker = "==>" if i == action else "   "
            print(f"{marker} {name:<12}: Q={q_values[i]:6.2f}")

        # Execute maneuver
        next_state, reward, done, truncated, step_info = env.step(action)
        state = next_state
        last_reward = reward
        total_reward += reward
        step += 1

        event = step_info.get("event", "MOVE_OK")
        print(
            f"\nExecuted: {action_name} | Outcome: {event} | "
            f"Reward: {reward:+.1f}"
        )

        if reward == -10.0:
            print("[!] WARNING: Agent entered high-risk Threat Zone!")
        elif reward == -5.0:
            print("[!] COLLISION: Agent impacted obstacle/perimeter!")

        time.sleep(delay_sec)

    # 3. Post-Mission Debriefing
    print("\n" + "=" * 55)
    print("               POST-MISSION DEBRIEF")
    print("=" * 55)
    print(f"Total Steps Elapsed : {step}")
    print(f"Cumulative Reward   : {total_reward:+.1f}")
    print(f"Final Agent Position: {env.agent_pos}")

    if last_reward == 100.0 or done:
        print("[+] STATUS: MISSION SUCCESS! Objective Secured.")
        success = True
    else:
        print("[-] STATUS: MISSION FAILED! Agent compromised or timed out.")
        success = False
    print("=" * 55)

    if render_mode == "gui":
        plt.ioff()
        plt.show()

    return success


if __name__ == "__main__":
    run_tactical_demo(render_mode="console", delay_sec=0.35)
