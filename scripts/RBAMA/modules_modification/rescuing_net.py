import argparse
from src.RBAMA import RBAMA
from src.RBAMA import rescuing_net
from src.environments import registered_versions

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="equip agent with new rescuing net")
    parser.add_argument('agent_name', type=str, help="Name of the agent")
    parser.add_argument('rescuing_net_name', type=str, help="Name of the rescuing net")

    args = parser.parse_args()
    agent_name = args.agent_name
    rescuing_net_name = args.rescuing_net_name

    agent, _ = RBAMA.setup_reasoning_agent(agent_name)
    agent.reasoning_unit.rescuing_net, _ = rescuing_net.setup_rescuing_agent(rescuing_net_name)
    agent_name = agent_name + "repl_resc"
    RBAMA.save_agent(agent, agent_name)