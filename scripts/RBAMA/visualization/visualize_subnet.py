import torch
from src.RBAMA import RBAMA
from src.RBAMA import rescuing_net
from src.RBAMA import guard_net
import argparse
import ast

def visualize_subnet(agent, env, state_reset=None, random_init="no randomness", seed=None):
        
        env.set_render_mode('human')
        agent.policy_dqn.eval()   

        while True:
            if state_reset:
                state, _ = env.reset(state=state_reset)
            else:
                state, _ = env.reset(random_init=random_init, seed=seed)
            terminated = False     
            truncated = False                  

            while(not terminated and not truncated): 
                if ("B" in env.get_lables() and isinstance(agent, guard_net.Guard)) or ("D" in env.get_lables() and isinstance(agent, rescuing_net.Rescuing)):
                    with torch.no_grad():
                        #take an action that is conform with the agent's moral obligations according to its current reason theory
                        dnn_input = agent.transformation(state)  
                        action_preferences = [tensor.item() for tensor in agent.policy_dqn(dnn_input)]
                        action = action_preferences.index(max(action_preferences))
                #if the subnet is not active, the agent idles
                else: 
                    action =5
                state,_,terminated,truncated,_ = env.step(action)


def main():

    parser = argparse.ArgumentParser(description="Visualize MORL agent behavior in the environment")
    
    parser.add_argument('agent_name', type=str, help="Name of the trained agent to load")
    parser.add_argument('subnet_type', type=str, help="Name of the subnet type to visualize ('rescuing', 'guard')")
    parser.add_argument('--state_reset', type=ast.literal_eval, default=None, help="Comma-separated values specifying the positions of the agent and each person on the flattened map, following the pattern: [agent_position, position_person_id_1, position_person_id_2, position_person_id_3, position_person_id_4]")
    parser.add_argument('--random_init', type=str, default="positions", help="Random initialization mode (default: 'positions'). Other values: 'no randomness', 'positions'")
    
    args = parser.parse_args()

    state_reset = args.state_reset
    
    agent, training_env = RBAMA.setup_reasoning_agent(args.agent_name)
    
    if args.subnet_type not in ['rescuing', 'waiting']:
        raise ValueError("Invalid subnet type. Choose either 'rescuing' or 'waiting'.")
    
    if args.subnet_type == 'rescuing':
        subnet= agent.reasoning_unit.rescuing_net
    elif args.subnet_type == 'guard':
        subnet= agent.reasoning_unit.guard_net

    visualize_subnet(subnet, training_env, state_reset=state_reset, random_init=args.random_init)


if __name__ == '__main__':
    main()
