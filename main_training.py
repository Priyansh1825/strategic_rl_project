
import time
import torch
import numpy as np
import os # Added for Windows terminal clearing
from tactical_env import TacticalGridEnv
from agent import DQNAgent

GRID_SIZE = 5
env = TacticalGridEnv(grid_size=GRID_SIZE)
agent = DQNAgent(grid_size=GRID_SIZE, action_size=env.action_space.n)

# 1. Load the fully trained brain
agent.model.load_state_dict(torch.load("strategic_agent_weights.pth"))
agent.model.eval() 

state, _ = env.reset()
done = False
step = 0
action_names = ["Move UP", "Move DOWN", "Move LEFT", "Move RIGHT"]

# Clear the Windows terminal
os.system('cls' if os.name == 'nt' else 'clear')
print("--- INITIATING TACTICAL DEMONSTRATION ---")
time.sleep(2) 

# 2. The Live Simulation Loop 
while not done and step < 20:
    # Clear the text log for a clean output
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Render the 2D Graph pop-up
    env.render()
    
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_values = agent.model(state_tensor)
    
    action = int(np.argmax(q_values.cpu().numpy()[0]))
    
    print(f"\n--- MISSION LOG ---")
    print(f"Step {step + 1}: Agent calculates optimal vector -> {action_names[action]}")
    
    state, reward, done, _, _ = env.step(action)
    
    if reward == -10:
        print("WARNING: Agent entered Threat Zone!")
    elif reward == -5:
        print("ERROR: Invalid maneuver (Obstacle hit).")
        
    step += 1

# Keep the final graph open at the end
import matplotlib.pyplot as plt
plt.ioff()
plt.show()

print(f"\n--- POST-MISSION REPORT ---")
if reward == 100:
    print("STATUS: MISSION SUCCESS. Objective Secured.")
else:
    print("STATUS: MISSION FAILED. Agent compromised or timed out.")