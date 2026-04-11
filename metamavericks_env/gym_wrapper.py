import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from metamavericks_env.server.metamavericks_env_environment import MetamavericksEnvironment
from metamavericks_env.models import MetamavericksAction

class MetamavericksGymWrapper(gym.Env):
    """
    Gymnasium wrapper for the Metamavericks OpenEnv environment.
    Compatible with Hugging Face ecosystem (stable-baselines3, rllib, etc.)
    """
    def __init__(self):
        super().__init__()
        self.env = MetamavericksEnvironment()
        
        # Action space: 6 floats, normalized to [-1.0, 1.0] for RL algorithms like PPO
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Observation space: 48 features, statically scaled to roughly [-1.0, 1.0] 
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(48,), dtype=np.float32)
        
        # Static scaling factors for each of the 16 features per agent (x3 agents = 48)
        self.scaling_factors = np.array([
            500.0, 500.0, # rel_pos
            15.0, 15.0,   # rel_vel
            15.0, 15.0,   # own_vel
            2.0, 2.0,     # ocean_curr
            500.0, 500.0, # haz_vec
            700.0,        # dist_to_haz
            700.0,        # dist_to_tar
            1.0, 1.0,     # norm_rel_pos
            1.0,          # agent_type
            1.0           # status_code
        ] * 3, dtype=np.float32)

    def _normalize_obs(self, features: list) -> np.ndarray:
        obs_array = np.array(features, dtype=np.float32)
        return np.clip(obs_array / self.scaling_factors, -10.0, 10.0)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        if options is None:
            options = {}
            
        # Call the OpenEnv reset
        obs_pydantic = self.env.reset(options)
        
        # Extract and normalize the features list
        obs_array = self._normalize_obs(obs_pydantic.features)
        
        # Use metadata as the initial info dictionary
        info = obs_pydantic.metadata
        
        return obs_array, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Un-normalize action from [-1, 1] to the true limits (Drone: 15.0, Boats: 8.0)
        action_scales = np.array([15.0, 15.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float32)
        scaled_action = np.clip(action, -1.0, 1.0) * action_scales
        
        # Convert numpy action array to Pydantic MetamavericksAction model
        agent_action = MetamavericksAction(commands=scaled_action.tolist())
        
        # Call the OpenEnv step
        obs_pydantic = self.env.step(agent_action)
        
        # Extract and normalize the features list
        obs_array = self._normalize_obs(obs_pydantic.features)
        
        # Update info with metadata
        info = obs_pydantic.metadata
        
        reward = obs_pydantic.reward
        done = obs_pydantic.done
        
        # Return observation, reward, terminated (done), truncated (False), info
        return obs_array, reward, done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.env.render(mode='rgb_array')
        return self.env.render()
