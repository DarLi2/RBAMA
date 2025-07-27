
from src.environments import registration
import gymnasium as gym
from collections import OrderedDict
import numpy as np
from gymnasium import spaces

"""wrapper for turning the observation into a multi-channel input to train CNNs"""
class Multi_Channel(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            len_obs_dict = len(self.get_obs_dict())
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len_obs_dict, self.bridge_map.height, self.bridge_map.width),  # (channels, height, width)
                dtype=np.float32
            )
        def observation(self, observation):
            obs_dict = self.get_obs_dict()
            multi_channel_input = multi_channel_input = np.stack([value for _, value in obs_dict.items()], axis=0)  # Stack along the channel axis (0)
            return multi_channel_input.astype(np.float32)

        def reset(self, seed=None, options=None, state = None, random_init= "no randomness"):
            observation, info = self.env.reset(seed=seed, options=options, state = state, random_init= random_init)
            observation = self.observation(observation)
            return observation, info
    