# Authored by Team MetaMavericks
import os
import sys
import gymnasium as gym

# Ensure the local environment package is discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), 'metamavericks_env'))

from stable_baselines3 import PPO
from metamavericks_env.gym_wrapper import MetamavericksGymWrapper

def train():
    print("Initializing AquaSAR-Env for RL training...")
    env = MetamavericksGymWrapper()
    
    print("Setting up PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log="./ppo_aquasar_tensorboard/"
    )
    
    total_timesteps = 300000
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    save_path = "ppo_aquasar_model"
    model.save(save_path)
    print(f"Training complete! Model saved to {save_path}.zip")
    
    env.close()

if __name__ == "__main__":
    train()
