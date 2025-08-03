import numpy as np
import random 
from collections import defaultdict
import gymnasium as gym
import numpy as np
from collections import defaultdict
from src.environments.state_space_estimation import state_space_estimation
from src.environments.bridge_person import Person_Moving, Person_Static

class Ethical_MOMDP_Wrapper(gym.Wrapper):
    def __init__(self, env, obs_config, e_w = 1):
        super().__init__(env)
        assert obs_config in ["CHVI", "scalarized", "None"], "invalid obs_config."
        self.obs_config = obs_config
        self.unwrapped.set_reward_type("MO")
        self.weights = [1,e_w]
        self.NOT_ON_MAP = self.to_1d_index(self.bridge_map.NOT_ON_MAP)

        self.states = defaultdict(set)
        if self.obs_config == "CHVI":
            self.states = state_space_estimation(self)

    def to_1d_index(self, position):
        position_1d = position[1] * self.bridge_map.width + position[0]
        return position_1d

    """
    returns an observation including the positions of the agent and the positions of all persons on the map (flattened to 1d);
    used for running the CHIV algorithm on the environment
    """
    def get_obs_CHVI(self):
        state = []
        agent_position = (self.env.get_agent_location()[1] * self.bridge_map.width) + self.env.get_agent_location()[0]
        state.append(agent_position)
        moving_person_positions = [self.NOT_ON_MAP for _ in range(4)] #initialize to "not_on_map"
        static_persons = [person for person in self.persons if isinstance(person, Person_Static)] #intitialize to "not_on_map"
        static_person_positions = [self.NOT_ON_MAP for _ in static_persons]
        person_positions = moving_person_positions + static_person_positions
        for person in self.persons:
            if isinstance(person, Person_Moving):
                person_positions[person.person_id-1] = self.to_1d_index(person.position)
            else:
                person_positions[4+static_persons.index(person)] = self.to_1d_index(person.position)
        state = state + person_positions
        return state
    
    def reset(self, seed=None, options=None, state = None, random_init = "positions"):
        observation, info = self.env.reset(state=state, random_init=random_init)
        if self.obs_config == "CHVI":
            observation = self.get_obs_CHVI()
        return observation, info 
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.obs_config == "None":
            return  observation, reward, terminated, truncated, info
        reward = [reward[0], reward[1] + reward[2]]
        if self.obs_config == "CHVI":
            return self.get_obs_CHVI(), reward, terminated, False, info
        elif self.obs_config == 'scalarized':
            reward = np.dot(self.weights, reward)
            return observation, reward, terminated, False, info
        
