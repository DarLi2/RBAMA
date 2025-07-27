from src.environments import registration
import gymnasium as gym
from collections import OrderedDict
import numpy as np
from gymnasium import spaces

"""wrapper for making the time left for a person until it drowns visible"""
class Drowning_Timer(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            len_obs_dict = len(self.get_obs_dict())
            total_grid_cells = self.env.bridge_map.width * self.env.bridge_map.height
            observation_size = len_obs_dict * total_grid_cells
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(observation_size,),
                dtype=np.integer
            )

        def normalized(self, value):
            return (self.env.drowning_time - value) / (self.env.drowning_time)

        def get_obs_dict(self):
            observation = self.env.get_obs_dict()
            grid_width = self.env.bridge_map.width
            grid_height = self.env.bridge_map.height
            timer_window = np.zeros((grid_height, grid_width)) 
            for y in range(grid_height):
                for x in range(grid_width):
                    for person in self.persons:
                        if self.bridge_map.in_water(person.position) and np.array_equal(person.position, np.array([x, y])):
                            timer_window[y, x] = self.normalized(person.drowning_behavior.time_in_water)  
            observation['timer_window'] = timer_window
            return observation
             
        def observation(self, observation):
            obs_dict = self.get_obs_dict()
            flattened_values = [window.flatten() for window in obs_dict.values()]
            nn_input = np.concatenate(flattened_values)
            return np.array(nn_input.astype(np.float32))

        def reset(self, seed=None, options=None, state = None, random_init= "no randomness"):
            observation, info = self.env.reset(seed=seed, options=options, state = state, random_init=random_init)
            observation = self.observation(observation)
            return observation, info
        
"""combines person window and dronwing-timer window"""
class Drowning_Timer_Combined(Drowning_Timer):
        def __init__(self, env):
            super().__init__(env)
            shape = self.observation_space.shape
            new_shape = (shape[0] - 1,)
            self.observation_space = spaces.Box(
                low=self.observation_space.low[0],
                high=self.observation_space.high[0],
                shape=new_shape,
                dtype=self.observation_space.dtype
            )
        def get_obs_dict(self):
            observation = self.env.get_obs_dict()
            grid_width = self.env.bridge_map.width
            grid_height = self.env.bridge_map.height
            for y in range(grid_height):
                for x in range(grid_width):
                    for person in self.persons:
                        if np.array_equal(person.position, np.array([x, y])):
                            if self.bridge_map.in_water(person.position):
                                observation["person_window"][y, x] = self.normalized(person.drowning_behavior.time_in_water)  # Mark person presence
                            else: 
                                observation["person_window"][y, x] = 1
            return observation
        
        

