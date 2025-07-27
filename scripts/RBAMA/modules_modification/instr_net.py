import argparse
from src.RBAMA import RBAMA
from src.RBAMA import rescuing_net
from src.environments import registered_versions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replace instrumental policy.")
    parser.add_argument('base_agent_name', type=str, help="Name of the agent to modify (recipient)")
    parser.add_argument('replacement_agent_name', type=str, help="Name of the agent whose policy DQN will be copied")

    args = parser.parse_args()
    base_agent_name = args.base_agent_name
    replacement_agent_name = args.replacement_agent_name

    base_agent, _ = RBAMA.setup_reasoning_agent(base_agent_name)
    replacement_agent, _ = RBAMA.setup_reasoning_agent(replacement_agent_name)

    base_agent.policy_dqn.load_state_dict(replacement_agent.policy_dqn.state_dict())

    new_agent_name = base_agent_name + "repl_instr"
    RBAMA.save_agent(base_agent, new_agent_name)

