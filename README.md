---
title: AquaSAR-Env
emoji: 🌊
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
---

# AquaSAR-Env: Autonomous Maritime Search & Rescue

### 🚀 Built by Team MetaMavericks for the Meta × PyTorch OpenEnv Hackathon

![Agent Simulation](eval_episode_test.gif)
*(Run `python run_app.py --test-rl` to generate and view this simulation GIF locally!)*

## Team Members

| Role | Name | Email |
| :--- | :--- | :--- |
| Team Lead | Vaghela Parthavsinh Ruchita | parthavsinh@gmail.com |
| Member | Krisha Patel | krishakpatel19@gmail.com |
| Member | Mansi Vora | mansivora279@gmail.com |

---

## Executive Summary

AquaSAR-Env is a high-impact, state-of-the-art multi-agent reinforcement learning environment engineered by **Team MetaMavericks**. Fully ported and integrated with the **OpenEnv standard**, it simulates a critical maritime search and rescue (SAR) operation coordinating one aerial drone (UAV) and two surface boats (USVs) to locate and rescue a drifting survivor in stochastic ocean conditions. 

Our solution focuses on autonomous coordination, dynamic pathfinding around hazards, and robust response to unpredictable environmental factors.

## Agent Roles & Behavior

The Reinforcement Learning model is trained to near-perfect precision using our advanced multi-agent reward shaping. 
- **The Drone (UAV):** Leverages its superior speed (15 m/s) to fly directly toward the lost survivor with perfect precision, acting as the primary scout and first responder.
- **Boat 1 (USV):** Follows the drone closely to secure the survivor for extraction.
- **Boat 2 (USV):** Keeps a close watch on the survivor and surrounding environment, remaining in formation to monitor for any secondary subjects or dynamic hazards during the rescue.

## Technical Innovations & OpenEnv Integration

Team MetaMavericks designed AquaSAR-Env to push the boundaries of autonomous maritime coordination:

- **Full OpenEnv Compatibility:** Native asynchronous WebSockets client (`metamavericks_env.client`) and a FastAPI server seamlessly support LLM inference and web dashboards.
- **Hugging Face RL Ecosystem Native:** The environment features a built-in `Gymnasium` wrapper (`metamavericks_env.gym_wrapper.MetamavericksGymWrapper`) that performs automatic Action and Observation scaling (scaling 48D physical bounds to `[-1, 1]`), fully optimizing it for Stable-Baselines3, RLlib, and TRL.
- **Advanced Multi-Agent Reward Shaping:** Utilizes a dense **Linear Distance Penalty** combined across all three agents to provide a constant, undeniable gradient toward the target. This completely eliminates random wandering and farming behaviors, ensuring perfect multi-agent convergence.
- **Heterogeneous Multi-Agent System**: Coordinates agents with different mobility constraints (UAVs at 15 m/s vs USVs at 8 m/s).
- **Stochastic Ocean Currents**: Implements realistic Ornstein-Uhlenbeck style random walk currents that affect the survivor's drift trajectory.
- **Dynamic Hazard Avoidance**: Incorporates active penalty zones (Rocky Shoals, -2.0 reward penalty) requiring agents to perform real-time pathfinding to avoid catastrophic failure.

## Environment Dynamics

### Observation Space
A 48-dimensional flat feature vector (16 features per agent), automatically scaled to roughly `[-1.0, 1.0]` by the wrapper:
- Relative position to target (x, y)
- Relative velocity to target (x, y)
- Own velocity (x, y)
- Ocean current vector (x, y)
- Nearest hazard vector (x, y)
- Scalar distance to nearest hazard
- Scalar distance to target
- Normalized relative position to target (x, y)
- Agent type (0.0 for drone, 1.0 for boat)
- Status code (0.0 for active, 1.0 for penalty)

### Action Space
A continuous 6-dimensional vector representing the $[v_x, v_y]$ velocity commands for the Drone, Boat 1, and Boat 2. The OpenEnv wrapper seamlessly accepts inputs in the range `[-1.0, 1.0]` and maps them to physical speeds (15.0 for Drone, 8.0 for Boats).

## How to Run the Application

We've provided a unified runner script `run_app.py` to handle all OpenEnv tasks.

### 1. Start the OpenEnv FastAPI Server (For Web UI / LLM Inference)
```bash
python run_app.py --server
```
*(Runs on port 8000. Leave this running in a separate terminal!)*

### 2. Run LLM Baseline Inference
Run this in a new terminal while the server is active.
```bash
# Set your Hugging Face API Token first
export HF_TOKEN="your_hugging_face_token_here"
# Or on Windows CMD: set HF_TOKEN="your_hugging_face_token_here"
# Or on PowerShell: $env:HF_TOKEN="your_hugging_face_token_here"

python run_app.py --inference
```
*(If you need help setting your token, run `python run_app.py --help-token`)*

### 3. Train the Reinforcement Learning Model
Trains the agent using Stable-Baselines3 (PPO) via the Gymnasium wrapper.
```bash
python run_app.py --train
```

### 4. Test the RL Model & Generate a GIF
Evaluates the trained agent and saves a visual simulation to `eval_episode_test.gif`.
```bash
python run_app.py --test-rl
```

---
*Engineered with precision by MetaMavericks.*
