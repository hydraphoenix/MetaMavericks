# Authored by Team MetaMavericks - Vaghela, Krisha, and Mansi

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import the custom Gymnasium wrapper we built for AquaSAR-Env
import sys
import os
from metamavericks_env.gym_wrapper import MetamavericksGymWrapper

def main():
    print("Initializing AquaSAR-Env for Reinforcement Learning training...")
    
    # 1. Instantiate the environment
    env = MetamavericksGymWrapper()
    
    print("Environment initialized successfully. Setting up PPO agent...")
    
    # 2. Initialize the PPO model with an MLP policy
    # Tweaked hyperparameters for ~10-15 mins on an i7 7th Gen laptop CPU
    # - Increased total_timesteps to 300,000 to allow the agent to fully narrow down its strategy.
    # - batch_size=256: Smoother gradient updates to help reduce the high Value Loss.
    # - ent_coef=0.005: Small entropy coefficient to carefully manage the high Entropy Loss, 
    #   encouraging it to exploit good actions while still exploring slightly.
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
    
    print("Starting training for 300,000 timesteps (Approx. 10-15 mins on i7 7th Gen)...")
    
    # 3. Train the model
    model.learn(total_timesteps=300000)
    
    # 4. Save the trained model
    # Stable Baselines3 automatically appends the .zip extension when saving
    save_path = "ppo_aquasar_model"
    model.save(save_path)
    print(f"Training complete! Model saved to {save_path}.zip")
    
    # Clean up the environment
    env.close()
    
    # 5. Evaluation Block
    print("\n--- Starting Evaluation Block ---")
    
    # Re-instantiate the environment
    eval_env = MetamavericksGymWrapper()
    eval_env.render_mode = "rgb_array"
    
    # Load the trained model
    trained_model = PPO.load(save_path)
    
    obs, info = eval_env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    
    frames = []
    
    while not done:
        # Pass the observation to the model to predict the next action
        action, _states = trained_model.predict(obs, deterministic=True)
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        frame = eval_env.render()
        if frame is not None:
            frames.append(frame)
            
        total_reward += reward
        steps += 1
        
        done = terminated or truncated
        
    print(f"Evaluation complete! Ran for {steps} steps.")
    print(f"Total Reward for 1 episode: {total_reward:.2f}")
    
    if frames:
        import imageio
        gif_path = "eval_episode.gif"
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"Saved evaluation video to {gif_path}")
        
    eval_env.close()

if __name__ == "__main__":
    main()
