import os
import sys
import json
import asyncio
from openai import OpenAI

# Add the environment path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'metamavericks_env'))
from metamavericks_env.client import MetamavericksEnv
from metamavericks_env.models import MetamavericksAction

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
# Strictly use injected environment variables for the validator
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY or "DUMMY_KEY"
)

def build_prompt(obs_features):
    prompt = "Here is the current situation:\n\n"
    agents = ["Drone", "Boat 1", "Boat 2"]
    
    for i in range(3):
        base_idx = i * 16
        rel_pos_x, rel_pos_y = obs_features[base_idx:base_idx+2]
        rel_vel_x, rel_vel_y = obs_features[base_idx+2:base_idx+4]
        own_vel_x, own_vel_y = obs_features[base_idx+4:base_idx+6]
        ocean_curr_x, ocean_curr_y = obs_features[base_idx+6:base_idx+8]
        haz_vec_x, haz_vec_y = obs_features[base_idx+8:base_idx+10]
        dist_to_haz = obs_features[base_idx+10]
        dist_to_tar = obs_features[base_idx+11]
        
        prompt += f"{agents[i]}:\n"
        prompt += f"  Relative position to survivor: ({rel_pos_x:.2f}, {rel_pos_y:.2f})\n"
        prompt += f"  Distance to survivor: {dist_to_tar:.2f}m\n"
        prompt += f"  Current velocity: ({own_vel_x:.2f}, {own_vel_y:.2f})\n"
        prompt += f"  Ocean current vector: ({ocean_curr_x:.2f}, {ocean_curr_y:.2f})\n"
        prompt += f"  Distance to nearest hazard: {dist_to_haz:.2f}m\n\n"
        
    return prompt

def get_action_from_llm(obs_features):
    system_prompt = "You are the Fleet Commander. Output EXACTLY a JSON list of 6 floats representing [vx, vy] for the Drone, Boat 1, and Boat 2. Max speed 15."
    user_prompt = build_prompt(obs_features)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        
        # Clean up markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        action_list = json.loads(content)
        
        if not isinstance(action_list, list) or len(action_list) != 6:
            print(f"[DEBUG] LLM returned invalid format: {content}", file=sys.stderr)
            return [0.0] * 6
            
        return [float(c) for c in action_list]
    except Exception as e:
        print(f"[DEBUG] Error communicating with LLM: {e}", file=sys.stderr)
        return [0.0] * 6

async def run_task_async(task_name):
    print(f"[START] task={task_name} env=metamavericks_env model={MODEL_NAME}", flush=True)
    
    # 1. Connect to the FastAPI server and RESET the environment using OpenEnv Client
    try:
        env_client = MetamavericksEnv(base_url=ENV_URL)
        res = await env_client.reset()
        obs_features = res.observation.features
    except Exception as e:
        print(f"[DEBUG] Failed to connect to server at {ENV_URL}: {e}", file=sys.stderr)
        print(f"[DEBUG] Did you forget to run `python run_app.py --server` in another terminal?", file=sys.stderr)
        return
    
    done = False
    step = 0
    rewards = []
    success = False
    
    while not done and step < 150:
        step += 1
        
        # We run the synchronous LLM call inside the async loop
        action_list = get_action_from_llm(obs_features)
        
        # 2. Step the server's environment so the Dashboard updates!
        try:
            action = MetamavericksAction(commands=action_list)
            step_result = await env_client.step(action)
        except Exception as e:
            print(f"[DEBUG] Failed to step environment at {ENV_URL}: {e}", file=sys.stderr)
            break
            
        obs_features = step_result.observation.features
        reward = step_result.reward or 0.0
        done = step_result.done
        info = step_result.observation.metadata
        
        rewards.append(reward)
        action_str = json.dumps(action_list).replace(" ", "")
        
        # [STEP] step=<n> action=<list> reward=<float> done=<bool> error=<null>
        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
        
        if done and info.get("min_boat_dist", float('inf')) <= 10.0:
            success = True
            
    if success:
        score = 0.999
    else:
        # We ensure score is never exactly 0.0 or 1.0 as per the validator's strict requirement.
        score = sum(rewards) / len(rewards) if rewards else 0.001
        score = max(0.001, min(0.999, score))
        
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    
    # [END] success=<bool> steps=<n> score=<float> rewards=<list>
    print(f"[END] success={str(success).lower()} steps={step} score={score:.3f} rewards={rewards_str}", flush=True)
    
    await env_client.close()

def main():
    tasks = ['easy', 'medium', 'hard']
    for task in tasks:
        asyncio.run(run_task_async(task))

if __name__ == "__main__":
    main()
