#!/usr/bin/env python3
import argparse
from src.MOBMA import MOBMA
from src.MOBMA.visualize import visualize
import ast


def main():
    parser = argparse.ArgumentParser(description="Visualize MORL agent behavior in the environment")
    
    parser.add_argument('agent_name', type=str, help="Name of the trained agent to load")
    parser.add_argument('--state_reset', type=ast.literal_eval, default=None, help="List of values specifying the positions of the agent and each person on the flattened map, following the pattern: [agent_position, position_person_id_1, position_person_id_2, position_person_id_3, position_person_id_4]")
    parser.add_argument('--random_init', type=str, default="positions", help="Random initialization mode (default: 'positions'). Common values: 'no randomness', 'positions'")
    
    args = parser.parse_args()
    
    agent, training_env = MOBMA.setup(args.agent_name)
        
    state_reset = args.state_reset

    visualize(agent=agent, env=training_env, state_reset=state_reset, random_init=args.random_init)

if __name__ == '__main__':
    main()
