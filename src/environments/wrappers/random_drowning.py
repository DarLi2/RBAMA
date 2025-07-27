from src.environments import registration
import gymnasium as gym
from collections import OrderedDict
import numpy as np
from gymnasium import spaces
from src.environments.bridge_person import Drowning_Random

"""wrapper for randomized drowning time"""
class Random_Drowning(gym.Wrapper):
        def __init__(self, env, prob):
            super().__init__(env)
            self.unwrapped.persons = []
            self.drowning_behavior = Drowning_Random(prob)
            self.initialize_persons(self.drowning_behavior)

        def reset(self, seed=None, options=None, state = None, random_init= "no randomness"):
            observation, info = self.env.reset(seed=seed, options=options, state = state, random_init= random_init)
            return observation, info
  
    