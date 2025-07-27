#!/usr/bin/env python3
from src.RBAMA import RBAMA
import argparse
from returns import eval_resoning_agent_returns
import ast

"""
evaluates the performance in terms of return values of an RBAMA on its training environment
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate an RBAMA agent in a selected environment.")

    parser.add_argument('agent_name', type=str, help="Name of the agent")
    parser.add_argument('episodes', type=int, help='Number of episodes the agent is evaluated on')
    parser.add_argument('--randomness', type=str, default='positions', help='Choose if (parts of) the state is reset randomly')
    parser.add_argument('--seed', type=int, help='Set a random seed')
    parser.add_argument('--state_reset', type=ast.literal_eval, help='Comma-separated values specifying the positions of the agent and each person on the flattened map, following the pattern: [agent_position, position_person_id_1, position_person_id_2, position_person_id_3, position_person_id_4]')

    args = parser.parse_args()
    agent_name = args.agent_name
    agent, agent_training_env = RBAMA.setup_reasoning_agent(agent_name)
    random_init = args.randomness
    seed = args.seed
    episodes = args.episodes
    state_reset = args.state_reset

    eval_resoning_agent_returns(agent, agent_training_env, episodes, state_reset=state_reset, random_init=random_init, seed=seed)
