import argparse
import subprocess
import sys
import os

def set_hf_token_help():
    print("\n" + "="*50)
    print("HUGGING FACE TOKEN SETUP INSTRUCTIONS")
    print("="*50)
    print("To run inference, you need a Hugging Face token (or alternative API key).")
    print("1. Get a token from: https://huggingface.co/settings/tokens")
    print("2. Set the environment variable in your terminal:")
    print("   - Windows (Command Prompt): set HF_TOKEN=your_token_here")
    print("   - Windows (PowerShell): $env:HF_TOKEN=\"your_token_here\"")
    print("   - Linux/macOS: export HF_TOKEN=\"your_token_here\"")
    print("="*50 + "\n")

def run_server():
    print("Starting the FastAPI server on http://localhost:8000...")
    print("You can view the live dashboard at http://localhost:8000/web")
    try:
        # Run using uv within metamavericks_env directory
        env_dir = os.path.join(os.path.dirname(__file__), 'metamavericks_env')
        subprocess.run(["uv", "run", "--project", env_dir, "server"], check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")

def train_model():
    print("Starting RL training...")
    try:
        subprocess.run([sys.executable, "train_rl.py"], check=True)
    except KeyboardInterrupt:
        print("\nTraining stopped.")

def test_rl_model():
    print("Testing the trained RL model...")
    if not os.path.exists("ppo_aquasar_model.zip"):
        print("Error: Trained model 'ppo_aquasar_model.zip' not found.")
        print("Please train the model first using: python run_app.py --train")
        return
        
    eval_script = """
import sys
import os
import imageio
from metamavericks_env.gym_wrapper import MetamavericksGymWrapper
from stable_baselines3 import PPO

print("\\n--- Starting Evaluation Block ---")
eval_env = MetamavericksGymWrapper()
eval_env.render_mode = "rgb_array"
try:
    trained_model = PPO.load("ppo_aquasar_model")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

obs, info = eval_env.reset()
done = False
total_reward = 0.0
steps = 0
frames = []

while not done:
    action, _states = trained_model.predict(obs, deterministic=True)
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
    gif_path = "eval_episode_test.gif"
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"Saved evaluation video to {gif_path}")
    
eval_env.close()
"""
    with open("temp_eval.py", "w") as f:
        f.write(eval_script)
    
    try:
        subprocess.run([sys.executable, "temp_eval.py"], check=True)
    finally:
        if os.path.exists("temp_eval.py"):
            os.remove("temp_eval.py")

def run_inference():
    print("Running LLM Baseline Inference (Requires Server to be running in another terminal)...")
    if not os.getenv("HF_TOKEN") and not os.getenv("API_KEY"):
        set_hf_token_help()
        print("Warning: HF_TOKEN or API_KEY environment variable is not set.")
        print("The inference script might fail or fallback to 0.0 actions if the API requires authentication.")
    
    try:
        subprocess.run([sys.executable, "inference.py"], check=True)
    except KeyboardInterrupt:
        print("\nInference stopped.")

def main():
    parser = argparse.ArgumentParser(description="AquaSAR-Env Project Runner - Team MetaMavericks")
    parser.add_argument("--server", action="store_true", help="Run the OpenEnv FastAPI environment server (needed for LLM inference and web dashboard)")
    parser.add_argument("--train", action="store_true", help="Train the RL model using Stable Baselines3 (PPO)")
    parser.add_argument("--test-rl", action="store_true", help="Test the trained RL model and generate a GIF")
    parser.add_argument("--inference", action="store_true", help="Run the LLM Baseline inference script (Requires server running)")
    parser.add_argument("--help-token", action="store_true", help="Show instructions on how to set a Hugging Face token")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        print("\nExamples:")
        print("  python run_app.py --train       # Trains the RL agent")
        print("  python run_app.py --test-rl     # Tests the trained RL agent and saves a GIF")
        print("  python run_app.py --server      # Starts the API server (for Web UI or LLM Inference)")
        print("  python run_app.py --inference   # Runs LLM evaluation (run server in a separate terminal first)")
        print("  python run_app.py --help-token  # Instructions on setting the HF token")
        return

    if args.help_token:
        set_hf_token_help()
    elif args.train:
        train_model()
    elif args.test_rl:
        test_rl_model()
    elif args.server:
        run_server()
    elif args.inference:
        run_inference()

if __name__ == "__main__":
    main()
