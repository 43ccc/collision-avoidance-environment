import gymnasium as gym
from pettingzoo import ParallelEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from gymnasium import spaces
from shapely.geometry import box
from copy import copy
import functools
from shapely.affinity import rotate

# TODO:
# - correct structure
# - able to properly import and such
# - add environment config
# - normalization stuff: obs, rew, action
# - change setting speed to acceleration
# - use max speed in reward function
# - allow for custom rewards functions
# - allow to manually set initial state
class Environment(ParallelEnv):
    metadata = {'render_modes': ['human'], 'render_fps': 20,
                'name': 'collision_avoidance_environment',}

    def __init__(self, num_agents=2, render_mode=None):
        self.size = 500
        self.agent_width = 5
        self.agent_length = 20
        self.target_reached_threshold = 1
        self.max_timesteps = 1000
        self.possible_agents = {f'agent_{i}' for i in range(num_agents)}

        self.observation_space = spaces.Dict(
            {
                'position': spaces.Box(0, self.size-1, shape=(2,), dtype=np.float32),
                'target': spaces.Box(0, self.size-1, shape=(2,), dtype=np.float32),
                'speed': spaces.Box(-10, 10, shape=(1,), dtype=np.float32),
                'direction': spaces.Box(0, 360, shape=(1,), dtype=np.float32)
            }
        )

        self.action_space = spaces.Box(low=np.array([0, -10]), high=np.array([360, 10]), shape=(2,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata['render_modes'], 'Invalid Render Mode'
        self.render_mode = render_mode
        self.rendering_initialized = False

    def _get_random_agent_states(self):
        return {agent_name: {k: self.observation_space[k].sample() for k in self.observation_space.keys()} for agent_name in self.possible_agents}

    def _init_human_render(self):
        self.fig, self.ax = plt.subplots()
        self.agent_patches = {}
        self.target_patches = {}
        
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)

        for i in range(self.num_agents):
            agent_name = f'agent_{i}'
            
            agent_patch = self._get_agent_patch(agent_name, i)
            target_patch = self._get_target_patch(agent_name, i)
            
            self.ax.add_patch(agent_patch)
            self.ax.add_patch(target_patch)

            self.agent_patches[agent_name] = agent_patch
            self.target_patches[agent_name] = target_patch

        plt.ion()

    # TODO: rename
    def _get_agent_xy(self, position):
        x,y = position
        x -= self.agent_width / 2
        y -= self.agent_length / 2

        return x, y

    def _get_target_patch(self, agent_name, idx, radius=5):
        x, y = self.agent_state[agent_name]['target']

        return Circle((x,y), radius=radius, facecolor=f'C{idx}')

    def _get_agent_patch(self, agent_name, idx):
        x, y = self._get_agent_xy(self.agent_state[agent_name]['position'])

        return Rectangle((x,y), self.agent_width, self.agent_length, angle=-self.agent_state[agent_name]['direction'], facecolor=f'C{idx}')

    def _get_obs(self):
        return {agent_name: {key: self.agent_state[agent_name][key] for key in self.observation_space} for agent_name in self.agents}
    
    def _get_info(self):
        return self.agent_info
    
    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.agent_state = self._get_random_agent_states()
        self.agent_info = {agent_name: {'distance_to_target': self._get_distance_to_goal(agent_name)} for agent_name in self.agents}

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _clip_agent_state(self, agent_state):
        for key in agent_state:
            agent_state[key] = np.clip(agent_state[key], a_min=self.observation_space[key].low, a_max=self.observation_space[key].high)     

    def _clip_actions(self, action):
        return np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)

    def _update_agent_state(self, action):
        for agent_name in self.agents:
            # Extract Actions
            clipped_action = self._clip_actions(action[agent_name])
            speed = clipped_action[0]
            direction_change = clipped_action[1]

            # Update agent state
            agent_state = self.agent_state[agent_name]
            agent_state['speed'] = np.array(speed)
            agent_state['direction'] = np.array((agent_state['direction'] + direction_change) % 360)
            
            heading = agent_state['direction'] - 90
            dx = agent_state['speed'] * np.cos(np.deg2rad(heading))
            dy = agent_state['speed'] * -np.sin(np.deg2rad(heading))
            agent_state['position'] += np.array([dx.item(), dy.item()])

            # Clip agent state to valid values
            self._clip_agent_state(agent_state)

    def _get_agent_hitbox(self, agent_name):
        x, y = self._get_agent_xy(self.agent_state[agent_name]['position'])

        agent_hitbox = box(x, y, x+self.agent_width, y+self.agent_length)
        agent_hitbox = rotate(geom=agent_hitbox, angle=-self.agent_state[agent_name]['direction'], origin='center')

        return agent_hitbox

    def _has_agent_collided(self, primary_agent):
        for secondary_agent in self.agents:
            if primary_agent != secondary_agent:
                primary_hitbox = self._get_agent_hitbox(primary_agent)
                secondary_hitbox = self._get_agent_hitbox(secondary_agent)
                
                if primary_hitbox.intersects(secondary_hitbox):
                    return True

        return False

    def _get_distance_to_goal(self, agent_name):
        return np.linalg.norm(self.agent_state[agent_name]['position'] - self.agent_state[agent_name]['target'], ord=2) # Euclidian distance

    def _has_agent_reached_goal(self, agent_name):
        return self.agent_info[agent_name]['distance_to_target'] <= self.target_reached_threshold
    
    def _get_distance_traveled_towards_goal(self, agent_name):
        last_distance = self.agent_info[agent_name]['distance_to_target']
        curr_distance = self._get_distance_to_goal(agent_name)

        return last_distance - curr_distance

    # TODO: I probably should initialize the agent info somewhere first
    def _update_agent_info(self):
        for agent_name in self.agents:
            self.agent_info[agent_name]['collided'] = self._has_agent_collided(agent_name)
            self.agent_info[agent_name]['distance_traveled_towards_target'] = self._get_distance_traveled_towards_goal(agent_name)
            self.agent_info[agent_name]['distance_to_target'] = self._get_distance_to_goal(agent_name)
            self.agent_info[agent_name]['reached_goal'] = self._has_agent_reached_goal(agent_name)

    def _get_terminations(self):
        return {agent_name: self.agent_info[agent_name]['reached_goal'] or self.agent_info[agent_name]['collided'] for agent_name in self.agents}

    def _get_truncations(self):
        if self.timestep > self.max_timesteps:
            return {agent_name: True for agent_name in self.agents}
        
        return {agent_name: False for agent_name in self.agents}

    # TODO: allow for custom reward functions
    def _get_rewards(self):
        rewards = {}

        for agent_name in self.agents:
            rewards[agent_name] = -0.1 # Negative base reward every step to favor actions

            if self.agent_info[agent_name]['collided']:
                rewards[agent_name] -= 1
            
            if self.agent_info[agent_name]['reached_goal']:
                rewards[agent_name] += 1
            
            rewards[agent_name] += 0.5 * self.agent_info[agent_name]['distance_traveled_towards_target'] / 10 # TODO: normalize with variable extracted from max speed

        return rewards

    def _update_agent_list(self, terminations, truncations):
        self.agents = []

        if not all(truncations.values()):
            for agent_name in terminations:
                if not terminations[agent_name]:
                    self.agents.append(agent_name)

    def step(self, action):
        self._update_agent_state(action)
        self._update_agent_info()
        terminations = self._get_terminations()
        truncations = self._get_truncations()
        rewards = self._get_rewards()
        self._update_agent_list(terminations, truncations)
        self.timestep += 1

        obs = self._get_obs()
        info = self._get_info()

        return obs, rewards, terminations, truncations, info
            
    def render(self):
        if self.render_mode == 'human':
            if not self.rendering_initialized:
                self._init_human_render()
                self.rendering_initialized = True

            for agent_name in self.agent_state:
                agent_state = self.agent_state[agent_name]
                self.agent_patches[agent_name].set_xy(self._get_agent_xy(agent_state['position']))
                self.agent_patches[agent_name].set_angle(-agent_state['direction'])

            plt.pause(1/self.metadata['render_fps'])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_space
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_space
