# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import uuid
import numpy as np
from typing import Dict, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MetamavericksAction, MetamavericksObservation
except ImportError:
    from models import MetamavericksAction, MetamavericksObservation


class MetamavericksEnvironment(Environment):
    """
    A multi-agent maritime search and rescue environment simulating UAV/USV coordination
    in stochastic ocean currents.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the metamavericks_env environment."""
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        
        self.grid_size = 500.0
        self.hazard_center = np.array([250.0, 250.0])
        self.hazard_radius = 40.0
        
        self.drone_max_speed = 15.0
        self.boat_max_speed = 8.0
        
        self.max_steps = 150
        self.rescue_dist = 10.0
        
        self.agent_positions = np.zeros((3, 2))
        self.agent_velocities = np.zeros((3, 2))
        self.agent_status = ['active'] * 3
        
        self.survivor_pos = np.zeros(2)
        self.survivor_vel = np.zeros(2)
        
        self.ocean_current = np.zeros(2)
        self.done = False
        self.reset()

    def reset(self, options: Dict[str, Any] = None) -> MetamavericksObservation:
        """Reset the environment."""
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self.done = False
        
        self.agent_positions = np.array([
            [250.0, 0.0],
            [200.0, 0.0],
            [300.0, 0.0]
        ])
        self.agent_velocities = np.zeros((3, 2))
        self.agent_status = ['active'] * 3
        
        self.survivor_pos = np.array([
            np.random.uniform(0, self.grid_size),
            np.random.uniform(300, self.grid_size)
        ])
        
        self.ocean_current = np.random.uniform(-1.0, 1.0, size=2)
        self.survivor_vel = self.ocean_current.copy()
        
        return self._get_observation()

    def step(self, action: MetamavericksAction) -> MetamavericksObservation:
        """Execute a step in the environment."""
        if self.done:
            return self._get_observation()
            
        self._state.step_count += 1
        
        cmds = np.array(action.commands).reshape(3, 2)
        
        # Apply velocity limits
        drone_speed = np.linalg.norm(cmds[0])
        if drone_speed > self.drone_max_speed:
            cmds[0] = cmds[0] / drone_speed * self.drone_max_speed
            
        for i in [1, 2]:
            boat_speed = np.linalg.norm(cmds[i])
            if boat_speed > self.boat_max_speed:
                cmds[i] = cmds[i] / boat_speed * self.boat_max_speed
                
        self.agent_velocities = cmds
        self.agent_positions += self.agent_velocities
        
        # Keep agents in bounds
        self.agent_positions = np.clip(self.agent_positions, 0.0, self.grid_size)
        
        # Update ocean current (random walk)
        self.ocean_current += np.random.normal(0.0, 0.1, size=2)
        self.ocean_current = np.clip(self.ocean_current, -2.0, 2.0)
        self.survivor_vel = self.ocean_current.copy()
        
        # Update survivor
        self.survivor_pos += self.survivor_vel
        self.survivor_pos = np.clip(self.survivor_pos, 0.0, self.grid_size)
        
        # Check hazard
        hazard_penalty = 0.0
        hazard_active = False
        self.agent_status = ['active'] * 3
        for i in range(3):
            dist_to_hazard = np.linalg.norm(self.agent_positions[i] - self.hazard_center)
            if dist_to_hazard < self.hazard_radius:
                hazard_penalty -= 2.0
                hazard_active = True
                self.agent_status[i] = 'hazard_penalty'

        # Calculate Reward
        dist_drone = np.linalg.norm(self.agent_positions[0] - self.survivor_pos)
        dist_boat1 = np.linalg.norm(self.agent_positions[1] - self.survivor_pos)
        dist_boat2 = np.linalg.norm(self.agent_positions[2] - self.survivor_pos)
        min_boat_dist = min(dist_boat1, dist_boat2)

        # Exponential decay flattens out at large distances, causing the agents to wander.
        # A linear distance penalty gives a constant, strong gradient towards the target.
        # We normalize by 500.0 (the grid size) to keep rewards well-scaled for PPO.
        avg_dist = (dist_drone + dist_boat1 + dist_boat2) / 3.0
        step_reward = - (avg_dist / 500.0)

        final_reward = float(step_reward + hazard_penalty)        
        if min_boat_dist <= self.rescue_dist:
            self.done = True
            final_reward += 10.0  # Big bonus for successfully rescuing
        elif self._state.step_count >= self.max_steps:
            self.done = True
            
        obs = self._get_observation()
        obs.reward = final_reward
        obs.done = self.done
        obs.metadata.update({
            'min_boat_dist': float(min_boat_dist),
            'hazard_active': hazard_active
        })
        
        return obs
        
    def _get_observation(self) -> MetamavericksObservation:
        """Get the current observation."""
        features = []
        for i in range(3):
            rel_pos = self.survivor_pos - self.agent_positions[i]
            rel_vel = self.survivor_vel - self.agent_velocities[i]
            own_vel = self.agent_velocities[i]
            ocean_curr = self.ocean_current
            haz_vec = self.hazard_center - self.agent_positions[i]
            dist_to_haz = np.linalg.norm(haz_vec)
            dist_to_tar = np.linalg.norm(rel_pos)
            norm_rel_pos = rel_pos / (dist_to_tar + 1e-6)
            agent_type = 0.0 if i == 0 else 1.0
            status_code = 1.0 if self.agent_status[i] == 'hazard_penalty' else 0.0
            
            features.extend(rel_pos.tolist())
            features.extend(rel_vel.tolist())
            features.extend(own_vel.tolist())
            features.extend(ocean_curr.tolist())
            features.extend(haz_vec.tolist())
            features.append(float(dist_to_haz))
            features.append(float(dist_to_tar))
            features.extend(norm_rel_pos.tolist())
            features.append(agent_type)
            features.append(status_code)
            
        metadata = {
            'task': 'search_and_rescue',
            'current_step': self._state.step_count,
            'wind_magnitude': float(np.linalg.norm(self.ocean_current)),
            'hazard_active': any(s == 'hazard_penalty' for s in self.agent_status)
        }
        
        return MetamavericksObservation(
            features=features,
            metadata=metadata,
            done=self.done,
            reward=0.0
        )

    @property
    def state(self) -> State:
        return self._state

    def render(self, mode='png'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        import imageio.v3 as iio
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_title(f"AquaSAR-Env | Step: {self._state.step_count}")
        
        hazard_circle = plt.Circle(self.hazard_center, self.hazard_radius, color='red', alpha=0.3, label='Rocky Shoal')
        ax.add_patch(hazard_circle)
        
        ax.scatter(self.agent_positions[0, 0], self.agent_positions[0, 1], c='green', marker='^', s=100, label='Drone')
        ax.scatter(self.agent_positions[1, 0], self.agent_positions[1, 1], c='blue', marker='s', s=80, label='Boat 1')
        ax.scatter(self.agent_positions[2, 0], self.agent_positions[2, 1], c='cyan', marker='s', s=80, label='Boat 2')
        
        ax.scatter(self.survivor_pos[0], self.survivor_pos[1], c='red', marker='*', s=150, label='Survivor')
        
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        if mode == 'rgb_array':
            return iio.imread(buf)
        return buf.read()
