import torch
from src.RBAMA import RBAMA
from src.RBAMA import rescuing_net
from src.RBAMA import guard_net
import argparse
import ast 

def visualize(agent, env, state_reset=None, random_init="no randomness", seed=None):

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
                morally_permissible_actions = agent.filter_actions(env, state)
                # select morally permissible action  
                with torch.no_grad():
                    #take an action that is conform with the agent's moral obligations according to its current reason theory
                    action_preferences = [tensor.item() for tensor in agent.policy_dqn(agent.transformation(state))]
                    action = action_preferences.index(max(action_preferences))
                    choice_list = list(action_preferences)
                    while action not in morally_permissible_actions:
                        choice_list.remove(max(choice_list))
                        action = action_preferences.index(max(choice_list))
                state,_,terminated,truncated,_ = env.step(action)          

def main():

    parser = argparse.ArgumentParser(description="Visualize RBAMA agent behavior in the environment")
    parser.add_argument('agent_name', type=str, help="Name of the trained agent to load")
    parser.add_argument('--state_reset', type=str, default=None, help="Comma-separated values specifying the positions of the agent and each person on the flattened map, following the pattern: [agent_position, position_person_id_1, position_person_id_2, position_person_id_3, position_person_id_4]")
    parser.add_argument('--random_init', type=str, default="positions", help="Random initialization mode (default: 'positions'). Common values: 'no randomness', 'positions'")
    
    args = parser.parse_args()
    
    agent, training_env = RBAMA.setup_reasoning_agent(args.agent_name)
        
    state_reset = args.state_reset
        
    visualize(agent=agent, env=training_env, state_reset=state_reset, random_init=args.random_init)

if __name__ == '__main__':
    main()
