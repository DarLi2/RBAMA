#!/usr/bin/env python3
import argparse
from src.MOBMA import MOBMA
import gymnasium as gym
from src.environments.wrappers.multi_channel import Multi_Channel
from src.environments.wrappers.random_drowning import Random_Drowning
from src.environments.wrappers.partial_observability import Partial_Observability

"""
b1_v1: s0: 0.53; every state initial: 3.56 (state = [35, 18, 49, 49, 34])
b1_v2: s0: 0.53; every state initial: 3.45 (state = [35, 18, 49, 49, 34])
b1_v1 random_drowning 0.5: s0: 1.4 
b2_v1: s0: 3.52; every state initial: 3.88
b2_v2: s0: 0.86, every state initial: 2.39 
bridge2_v3_base: s0: 0.91; every state initial: 1.95
bridge2_v3_ds1: s0: 0.86, every state initial: 2.2
"""

def main():
    parser = argparse.ArgumentParser(description="Train a MOBMA.")

    parser.add_argument('env_id', type=str, help='Gym environment ID')
    parser.add_argument('training_episodes', type=int)
    parser.add_argument('ethical_weight', type=float)
    parser.add_argument('--use_CNN', action='store_true', help='Use a CNN')
    parser.add_argument('--use_random_drowning', action='store_true', help='Apply the Random_Drowning wrapper')
    parser.add_argument('--use_partial_observability', action='store_true', help='Apply the Partial_observability wrapper')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    agent = MOBMA.DRL_agent(env=env, e_w=args.ethical_weight)
    ## training
    agent_name =  args.env_id + "Episodes" + str(args.training_episodes) + "wE" + str(args.ethical_weight)
    agent.train(args.training_episodes, agent_name)
    MOBMA.save_agent(agent, agent_name)

if __name__ == "__main__":
    main()