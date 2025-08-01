import argparse
from src.RBAMA import RBAMA
from src.RBAMA import guard_net
from src.environments import registered_versions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Equip an agent with a new waiting net.")
    parser.add_argument('agent_name', type=str, help="Name of the agent to modify")
    parser.add_argument('guard_net_name', type=str, help="Name of the guard net to use")

    args = parser.parse_args()
    agent_name = args.agent_name
    guard_net_name = args.guard_net_name

    # Load the target agent
    agent, _ = RBAMA.setup_reasoning_agent(agent_name)

    # Load and assign the guard net
    agent.reasoning_unit.guard_net, _ = guard_net.setup_on_bridge(guard_net_name)

    # Save updated agent with new name
    updated_agent_name = agent_name + "wait_repl"
    RBAMA.save_agent(agent, updated_agent_name)
