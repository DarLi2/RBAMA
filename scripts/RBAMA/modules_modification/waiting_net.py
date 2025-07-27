import argparse
from src.RBAMA import RBAMA
from src.RBAMA import waiting_net
from src.environments import registered_versions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Equip an agent with a new waiting net.")
    parser.add_argument('agent_name', type=str, help="Name of the agent to modify")
    parser.add_argument('waiting_net_name', type=str, help="Name of the waiting net to use")

    args = parser.parse_args()
    agent_name = args.agent_name
    waiting_net_name = args.waiting_net_name

    # Load the target agent
    agent, _ = RBAMA.setup_reasoning_agent(agent_name)

    # Load and assign the waiting net
    agent.reasoning_unit.waiting_net, _ = waiting_net.setup_on_bridge(waiting_net_name)

    # Save updated agent with new name
    updated_agent_name = agent_name + "wait_repl"
    RBAMA.save_agent(agent, updated_agent_name)