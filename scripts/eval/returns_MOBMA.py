#!/usr/bin/env python3
from src.MOBMA import MOBMA
import argparse
from returns import eval_MORL_agent
import ast

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a MOBMA agent in a selected environment.")

    parser.add_argument('agent_name', type=str, help="Name of the agent")
    parser.add_argument('episodes', type=int, help='Number of episodes the agent is evaluated on')
    parser.add_argument('--randomness', type=str, default='positions', help='Choose if (parts of) the state is reset randomly')
    parser.add_argument('--seed', type=int, help='Set a random seed')
    parser.add_argument('--state_reset', type=ast.literal_eval, help='List of values specifying the positions of the agent and each person on the flattened map, following the pattern: [agent_position, position_person_id_1, position_person_id_2, position_person_id_3, position_person_id_4]')

    args = parser.parse_args()
    agent_name = args.agent_name
    agent, agent_training_env = MOBMA.setup(args.agent_name)
    episodes = args.episodes
    random_init = args.randomness
    state_reset = args.state_reset
    seed = args.seed

    eval_MORL_agent(agent=agent, env=agent_training_env, n_episodes=episodes,state_reset=state_reset,random_init=random_init, seed=seed)