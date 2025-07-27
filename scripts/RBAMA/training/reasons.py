from src.RBAMA import RBAMA
from src.environments import registered_versions
from src.RBAMA import moral_judge
from src.RBAMA import translator
import networkx as nx
import argparse
from scripts.utils import create_judge
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
    
    parser = argparse.ArgumentParser(description="Train the reasoning theory of the RBAMA.")
    parser.add_argument('agent_name', type=str, help='Name of the agent')
    parser.add_argument('training_episodes', type=int)
    parser.add_argument('judge', type=str, default='prioR', choices=['prioR', 'prioW', 'onlyR', 'onlyW'],
                        help='Type of moral judge to use (default: prioR)')
    parser.add_argument('--no_randomness', action='store_true', help='Sets agent to the upper left corner of the map at the beginning of each episode')
    
    args = parser.parse_args()

    judge = create_judge(args.judge)
    agent, agent_training_env = RBAMA.setup_reasoning_agent(args.agent_name)
    agent.reasoning_unit.G = nx.DiGraph()
    agent_training_env.set_render_mode(None)

    if args.no_randomness:
        random_init = "no_randomness"
    else:
        random_init = "positions"

    agent.train_reasons(
        episodes=args.training_episodes,
        env=agent_training_env,
        judge=judge,
        random_init=random_init
    )

    updated_agent_name = f"{args.agent_name}Reas{args.training_episodes}"
    RBAMA.save_agent(agent, updated_agent_name)
