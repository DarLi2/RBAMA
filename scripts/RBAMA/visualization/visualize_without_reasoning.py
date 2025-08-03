import torch
from src.RBAMA import RBAMA
from src.RBAMA import rescuing_net
from src.RBAMA import guard_net
import argparse
import ast

def visualize_without_reasoning(agent, env, state_reset=None, random_init="no randomness"):

        env.set_render_mode('human')
        agent.policy_dqn.eval()   

        while True:
            if state_reset:
                state, _ = env.reset(state=state_reset)
            else:
                state, _ = env.reset(random_init=random_init)
            terminated = False     
            truncated = False                  

            while(not terminated and not truncated): 
                with torch.no_grad():
                    #take the preferred action
                    action = agent.policy_dqn(agent.transformation(state)).argmax().item()
                state,_,terminated,truncated,_ = env.step(action)


def main():

    parser = argparse.ArgumentParser(description="Visualize MORL agent behavior in the environment")
    
    parser.add_argument('agent_name', type=str, help="Name of the trained agent to load")
    parser.add_argument('--state_reset', type=ast.literal_eval, default=None, help="List of values specifying the positions of the agent and each person on the flattened map, following the pattern: [agent_position, position_person_id_1, position_person_id_2, position_person_id_3, position_person_id_4]")
    parser.add_argument('--random_init', type=str, default="positions", help="Random initialization mode (default: 'positions'). Other values: 'no randomness', 'positions'")
    
    args = parser.parse_args()

    state_reset = args.state_reset
    
    agent, training_env = RBAMA.setup_reasoning_agent(args.agent_name)

    visualize_without_reasoning(agent, training_env, state_reset=state_reset, random_init=args.random_init)

if __name__ == '__main__':
    main()