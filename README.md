# Tactical Strategic RL // Digital Twin Mission Navigation Engine

Autonomous spatial decision-making agent powered by PyTorch Deep Q-Networks (DQN). The project features a full interactive Digital Twin Operator Console modeled after the enterprise architecture of `Industrial_MultiCamera_Tracking`, complete with high-FPS real-time canvas rendering, action Q-value radar meters, and a live telemetry mission stream.

---

## Architecture Overview

```
[TacticalGridEnv (5x5)] ──> [StrategicDQN (PyTorch)] ──> [Action Decision Policy]
        │                              │                              │
        │ Coordinates                  │ Q-Values                     │ UP, DOWN, LEFT, RIGHT
        ▼                              ▼                              ▼
 [FastAPI REST / WS] ──> [Live WebSocket Stream] ──> [Digital Twin Operator Console]
                                                        ├── 2D Mission Grid Canvas
                                                        ├── Q-Value Action Radar
                                                        └── Live Mission Telemetry Log
```

---

## Key Features

1. **Autonomous Spatial Navigation**: Deep Q-Learning agent with experience replay, target network synchronization, and Bellman optimality updates.
2. **Hazard Avoidance**: Optimal path planning dodging threat zones (`-10.0` penalty) and physical obstacles (`-5.0` collision penalty).
3. **Digital Twin Operator Console (`src/dashboard/index.html`)**:
   - Cyber-tactical dark mode interface with real-time vector trail.
   - Live **Action Q-Value Radar** visualizing relative model confidence across `UP`, `DOWN`, `LEFT`, and `RIGHT`.
   - Real-time step-by-step telemetry, step rewards, and cumulative score.
   - Interactive controls: Step AI, Auto-Run, Reset, and simulation speed adjustment.
4. **FastAPI & WebSocket Backend (`src/api/server.py`)**:
   - High-throughput REST endpoints (`/api/status`, `/api/reset`, `/api/step`, `/api/run-mission`).
   - Bidirectional WebSocket stream (`/ws/live`) for real-time mission synchronization.

---

## Quick Start

### 1. Launch Interactive Digital Twin Operator Console
Double-click `run_server.bat` or run:
```powershell
python run_demo.py
```
This automatically boots the FastAPI backend and opens `http://127.0.0.1:8000` in your default browser.

### 2. Run CLI Mission Demonstration
```powershell
python main_training.py
```

### 3. Re-train Policy Weights
```powershell
python train.py 350
```

---

## File Structure

```
strategic_rl_project/
├── agent.py                      # DQNAgent with experience replay and target network
├── model.py                      # StrategicDQN 4-layer fully-connected architecture
├── tactical_env.py               # TacticalGridEnv with hazard detection and reward shaping
├── train.py                      # Multi-episode training harness with metric logging
├── main_training.py              # CLI evaluation and mission telemetry debrief
├── run_demo.py                   # Automated web operator console launcher
├── run_server.bat                # Windows one-click batch launcher
├── strategic_agent_weights.pth   # Pre-trained checkpoint (100% mission win rate)
└── src/
    ├── api/
    │   └── server.py             # FastAPI REST & WebSocket streaming server
    └── dashboard/
        └── index.html            # High-FPS Digital Twin operator HUD console
```

---

## QA & Code Quality

Verify PEP8 compliance:
```powershell
python -m flake8 .
```
All modules adhere strictly to PEP8 standards (0 errors).
