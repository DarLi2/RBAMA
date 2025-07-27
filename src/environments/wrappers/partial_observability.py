from src.environments import registration
import gymnasium as gym
from collections import OrderedDict
import numpy as np
from gymnasium import spaces

"""wrapper for restricting the agent's observation window such that it does not obeserve the whole map"""
class Partial_Observability(gym.ObservationWrapper):
    def __init__(self, env, obs_window_size):
        super().__init__(env)
        self.obs_window_size = obs_window_size
        len_obs_dict = len(self.get_obs_dict())
        self.observation_space_size = len_obs_dict* ((2*self.obs_window_size+1) ** 2)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.observation_space_size,),
            dtype=np.integer
        )
    
    def observation(self, observation):
        # define observation window size
        half_window_size = self.obs_window_size
        window_size = 2 * half_window_size + 1

        # agent's current position
        agent_x, agent_y = self.get_agent_location()

        # calculate the top-left corner of the observation window
        cutout_top_left_x = max(0, agent_x - half_window_size)
        cutout_top_left_y = max(0, agent_y - half_window_size)

        # adjust the window if it exceeds grid bounds
        if cutout_top_left_x + window_size > self.bridge_map.width:
            cutout_top_left_x = self.bridge_map.width - window_size
        if cutout_top_left_y + window_size > self.bridge_map.height:
            cutout_top_left_y = self.bridge_map.height - window_size

        # calculate the bottom-right corner of the observation window
        cutout_bottom_right_x = cutout_top_left_x + window_size
        cutout_bottom_right_y = cutout_top_left_y + window_size

        # get the full grid observation using get_obs_dict()
        full_observation = self.get_obs_dict()

        observation_windows = OrderedDict()

        # extract observation windows dynamically
        for key, full_grid in full_observation.items():
            # initialize an empty window for the current grid
            window = np.zeros((window_size, window_size))

            # extract the relevant slice from the full grid
            for y in range(cutout_top_left_y, cutout_bottom_right_y):
                for x in range(cutout_top_left_x, cutout_bottom_right_x):
                    # Get relative coordinates for the window
                    relative_x = x - cutout_top_left_x
                    relative_y = y - cutout_top_left_y

                    # Copy value from the full grid to the observation window
                    window[relative_y, relative_x] = full_grid[y, x]
            
            # store the window dynamically in the dictionary
            observation_windows[key] = window

        # flatten and concatenate the observation windows dynamically
        flattened_values = [window.flatten() for window in observation_windows.values()]
        nn_input = np.concatenate(flattened_values)

        return np.array(nn_input.astype(np.float32))

    def reset(self, seed=None, options=None, state = None, random_init= "no randomness"):
            observation, info = self.env.reset(seed=seed, options=options, state = state, random_init=random_init)
            observation = self.observation(observation=observation)
            return observation, info
    