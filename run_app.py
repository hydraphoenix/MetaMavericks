import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="AquaSAR-Env Unified Runner")
    parser.add_argument("--server", action="store_true", help="Start the OpenEnv FastAPI server")
    parser.add_argument("--inference", action="store_true", help="Run LLM baseline inference")
    parser.add_argument("--train", action="store_true", help="Train the RL model")
    parser.add_argument("--test-rl", action="store_true", help="Test the RL model and generate GIF")
    parser.add_argument("--help-token", action="store_true", help="Help with setting the HF_TOKEN")

    args = parser.parse_args()

    if args.help_token:
        print("\nTo set your Hugging Face API Token:")
        print("Windows CMD: set HF_TOKEN=your_token_here")
        print("Windows PowerShell: $env:HF_TOKEN='your_token_here'")
        print("Linux/macOS: export HF_TOKEN='your_token_here'\n")
        return

    if args.server:
        print("Starting OpenEnv FastAPI server on port 8000...")
        # Run the server module
        subprocess.run([sys.executable, "-m", "server.app"])

    elif args.inference:
        print("Running LLM baseline inference...")
        subprocess.run([sys.executable, "inference.py"])

    elif args.train:
        print("Training Reinforcement Learning model...")
        subprocess.run([sys.executable, "train_rl.py"])

    elif args.test_rl:
        print("Testing RL model...")
        # Assuming train_rl.py has an evaluation block that saves a gif
        subprocess.run([sys.executable, "train_rl.py"])

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
