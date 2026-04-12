import os
import sys
import argparse
import asyncio
import imageio
import numpy as np

# Ensure the local environment package is discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), 'metamavericks_env'))

from stable_baselines3 import PPO
from metamavericks_env.gym_wrapper import MetamavericksGymWrapper

def evaluate_rl():
    print("\n--- Evaluating Trained RL Model ---")
    model_path = "ppo_aquasar_model"
    if not os.path.exists(f"{model_path}.zip"):
        print(f"Error: Model {model_path}.zip not found. Run 'python train.py' first.")
        return

    env = MetamavericksGymWrapper()
    env.render_mode = "rgb_array"
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    frames = []

    print("Running episode...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)
            
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
    print(f"Evaluation complete! Steps: {steps}, Total Reward: {total_reward:.2f}")

    if frames:
        gif_path = "eval_episode_test.gif"
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"Saved simulation to {gif_path}")
    
    env.close()

async def evaluate_llm():
    print("\n--- Running LLM Inference (Requires Server) ---")
    # This imports and runs the logic from inference.py
    from inference import run_task_async
    await run_task_async("eval")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MetaMavericks Model")
    parser.add_argument("--mode", choices=["rl", "llm"], default="rl", help="Evaluation mode")
    args = parser.parse_args()
    
    if args.mode == "rl":
        evaluate_rl()
    else:
        asyncio.run(evaluate_llm())
