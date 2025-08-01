from src.RBAMA import RBAMA
from src.environments.wrappers.multi_channel import Multi_Channel
from src.environments.wrappers.random_drowning import Random_Drowning
from src.environments.wrappers.partial_observability import Partial_Observability
import gymnasium as gym
import argparse
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
    parser = argparse.ArgumentParser(description="Train an RBAMA agent in a selected environment with optional wrappers.")

    parser.add_argument('env_id', type=str, help='Gym environment ID')
    parser.add_argument('training_episodes', type=int)
    parser.add_argument('--use_CNN', action='store_true', help='Use a CNN')
    parser.add_argument('--use_random_drowning', action='store_true', help='Apply the Random_Drowning wrapper')
    parser.add_argument('--use_partial_observability', action='store_true', help='Apply the Partial_observability wrapper')

    args = parser.parse_args()

    # Setup environment
    env = gym.make(args.env_id)

    if args.use_CNN:
        env = Multi_Channel(env)
        agent = RBAMA.RBAMA_CNN(env)
    else:
        agent = RBAMA.RBAMA(env)
    if args.use_random_drowning:
        env = Random_Drowning(env)
    if args.use_partial_observability:
        env = Partial_Observability(env)

    agent_name = f"{args.env_id}_instr_{args.training_episodes}"
    agent.train(args.training_episodes, agent_name, env)
    RBAMA.save_agent(agent, agent_name)
