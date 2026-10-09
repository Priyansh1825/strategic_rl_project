import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from agent import DQNAgent
from tactical_env import TacticalGridEnv

app = FastAPI(
    title="Tactical Strategic RL Live Operator Console",
    description="Live RL Mission Dashboard with Real-Time Q-Value Radar.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_PATH = str(PROJECT_ROOT / "strategic_agent_weights.pth")
DASHBOARD_HTML = PROJECT_ROOT / "src" / "dashboard" / "index.html"

# Global environment and agent singletons
grid_size = 5
env = TacticalGridEnv(grid_size=grid_size, max_steps=25)
agent = DQNAgent(grid_size=grid_size, action_size=env.action_space.n)

if os.path.exists(WEIGHTS_PATH):
    agent.load(WEIGHTS_PATH)
    print(f"[+] Server loaded weights from: {WEIGHTS_PATH}")
else:
    print(f"[!] Warning: No weights file found at: {WEIGHTS_PATH}")


class MapConfig(BaseModel):
    agent_pos: Optional[List[int]] = None
    target_pos: Optional[List[int]] = None
    threats: Optional[List[List[int]]] = None
    obstacles: Optional[List[List[int]]] = None


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the Tactical RL Operator Live Dashboard."""
    if not DASHBOARD_HTML.exists():
        raise HTTPException(
            status_code=404, detail="Dashboard index.html not found"
        )
    with open(DASHBOARD_HTML, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Retrieves operational telemetry and model specs."""
    return {
        "status": "ONLINE",
        "device": str(agent.device),
        "grid_size": env.grid_size,
        "weights_loaded": os.path.exists(WEIGHTS_PATH),
        "agent_pos": env.agent_pos,
        "target_pos": list(env.target_pos),
        "threats": [list(t) for t in env.threats],
        "obstacles": [list(o) for o in env.obstacles],
        "step_count": env.step_count,
    }


@app.post("/api/reset")
def reset_mission(config: Optional[MapConfig] = None) -> Dict[str, Any]:
    """Resets the tactical map with optional custom configuration."""
    start = tuple(config.agent_pos) if config and config.agent_pos else None
    target = (
        tuple(config.target_pos) if config and config.target_pos else None
    )
    threats = (
        [tuple(t) for t in config.threats]
        if config and config.threats else None
    )
    obstacles = (
        [tuple(o) for o in config.obstacles]
        if config and config.obstacles else None
    )

    state, info = env.reset(
        start_pos=start,
        target_pos=target,
        threats=threats,
        obstacles=obstacles,
    )
    q_vals = agent.get_q_values(state)
    best_action = int(agent.act(state, evaluate=True))

    return {
        "state": state.tolist(),
        "info": info,
        "q_values": q_vals,
        "recommended_action": best_action,
        "action_name": TacticalGridEnv.ACTION_NAMES[best_action],
    }


@app.post("/api/step")
def step_mission(action: Optional[int] = None) -> Dict[str, Any]:
    """Executes single step using model inference if action is None."""
    curr_state = env.grid.copy()

    if action is None:
        action = int(agent.act(curr_state, evaluate=True))

    next_state, reward, done, truncated, info = env.step(action)
    next_q_vals = agent.get_q_values(next_state)
    next_best_action = int(agent.act(next_state, evaluate=True))

    return {
        "action_taken": action,
        "action_name": TacticalGridEnv.ACTION_NAMES[action],
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "info": info,
        "state": next_state.tolist(),
        "q_values": next_q_vals,
        "next_recommended_action": next_best_action,
        "next_action_name": TacticalGridEnv.ACTION_NAMES[next_best_action],
    }


@app.post("/api/run-mission")
def run_full_mission() -> Dict[str, Any]:
    """Runs autonomous simulation and returns complete trajectory."""
    state, info = env.reset()
    trajectory = []
    total_reward = 0.0
    done = False
    truncated = False

    while not done and not truncated:
        q_vals = agent.get_q_values(state)
        action = int(agent.act(state, evaluate=True))
        action_name = TacticalGridEnv.ACTION_NAMES[action]

        step_record = {
            "step": env.step_count + 1,
            "agent_pos": list(env.agent_pos),
            "action": action,
            "action_name": action_name,
            "q_values": q_vals,
        }

        next_state, reward, done, truncated, step_info = env.step(action)
        step_record["reward"] = reward
        step_record["next_agent_pos"] = list(env.agent_pos)
        step_record["event"] = step_info.get("event", "MOVE_OK")
        total_reward += reward
        trajectory.append(step_record)
        state = next_state

    return {
        "success": bool(done and total_reward > 50.0),
        "total_steps": len(trajectory),
        "total_reward": total_reward,
        "final_pos": env.agent_pos,
        "trajectory": trajectory,
    }


@app.websocket("/ws/live")
async def live_mission_stream(websocket: WebSocket):
    """Real-time WebSocket streaming of live autonomous mission execution."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            action_type = cmd.get("type", "START")

            if action_type == "START":
                delay = float(cmd.get("delay", 0.3))
                state, _ = env.reset()
                done = False
                truncated = False

                while not done and not truncated:
                    q_vals = agent.get_q_values(state)
                    action = int(agent.act(state, evaluate=True))
                    next_state, reward, done, truncated, info = env.step(
                        action
                    )

                    payload = {
                        "type": "STEP_UPDATE",
                        "step": env.step_count,
                        "agent_pos": env.agent_pos,
                        "action": action,
                        "action_name": TacticalGridEnv.ACTION_NAMES[action],
                        "reward": reward,
                        "done": done,
                        "truncated": truncated,
                        "event": info.get("event", "MOVE_OK"),
                        "q_values": q_vals,
                        "grid": next_state.tolist(),
                    }
                    await websocket.send_text(json.dumps(payload))
                    state = next_state
                    await asyncio.sleep(delay)

                final_payload = {
                    "type": "MISSION_COMPLETE",
                    "success": bool(done and reward == 100.0),
                    "total_steps": env.step_count,
                    "final_pos": env.agent_pos,
                }
                await websocket.send_text(json.dumps(final_payload))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app", host="127.0.0.1", port=8000, reload=False
    )
