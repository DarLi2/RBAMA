import random 
from collections import defaultdict

"""
returns an estimation of the state space through random sampling (needed for CHVI)
"""
def state_space_estimation(env):
    states =  defaultdict(set)
    episodes = 3000
    for _ in range(episodes):
        observation, _ = env.reset(random_init = "positions")  
        terminated = False      
        truncated = False   
        while not terminated and not truncated:
            for i in range(len(observation)):
                    states[i].add(observation[i])
            action = random.choice(list(range(5)))
            observation,_,terminated,truncated,_ = env.step(action)
    env.reset(random_init = "positions")
    return states