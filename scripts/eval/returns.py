import numpy as np
import torch
from src.environments import registered_versions, bridge_person
from tqdm import tqdm

"""evaluation in terms of return values for the RBAMA; also counts the number of times the agent solves the moral task by prioritizing pushing a person off the bridge over rescuing a drowning person"""

def eval_resoning_agent_returns(agent, env, n_episodes, state_reset=None, random_init="no randomness", seed=None):
    agent.policy_dqn.eval()  
    agent.reasoning_unit.guard_net.policy_dqn.eval()
    agent.reasoning_unit.rescuing_net.policy_dqn.eval() 
    reward_instr = 0
    reward_resc = 0
    reward_wait = 0
    pushing_lower_order_sum = 0
    in_water_sum = 0
    env.set_reward_type("MO")
    for _ in tqdm(range(n_episodes)):
        in_water = False
        if state_reset:
            state, _ = env.reset(state=state_reset)
        else:
            state, _ = env.reset(random_init=random_init, seed=seed)
        terminated = False     
        truncated = False   

        while(not terminated and not truncated): 
            for person in env.persons:
                if person.status == bridge_person.Status.IN_WATER_AT_RISK:
                    in_water = True   
            morally_permissible_actions = agent.filter_actions(env, state)
            # if there is only one morally permissible action, bypass the NN that optimizes for achieving the instrumental policy and execute the action directly (always the case when moral task takes priority)
            if len(morally_permissible_actions) == 1:
                action = morally_permissible_actions[0]
            # otherwise, select the action that is morally conform with all duties and has the highest expecet instrumental value
            else:
                with torch.no_grad():
                    action_preferences = [tensor.item() for tensor in agent.policy_dqn(agent.transformation(state))]
                action = action_preferences.index(max(action_preferences))
                choice_list = list(action_preferences)
                while action not in morally_permissible_actions:
                    choice_list.remove(max(choice_list))
                    action = action_preferences.index(max(choice_list))

            pushing_lower_order = False
            for person in env.persons:
                if env.bridge_map.position_on_bridge(person.position):
                    adjacent_tiles = env.bridge_map.get_adjacent_tiles(env.get_agent_location())
                    if any(np.array_equal(person.position, tile) for tile in adjacent_tiles):
                        if "D" in env.get_lables() and 'C' not in agent.reasoning_unit.moral_obligations(env.get_lables(), state):
                            pushing_lower_order = True    

            state,reward,terminated,truncated,_ = env.step(action)

            reward_instr += reward[0]
            reward_wait  += reward[1]
            reward_resc  += reward[2]

            if pushing_lower_order:
                pushing_lower_order_sum += 1

        if in_water:
            in_water_sum += 1

    print("reward instr:", reward_instr, "reward push:", reward_wait, "reward resc:", reward_resc, "in_water_sum:", in_water_sum, "pushing_lower_order_sum:", pushing_lower_order_sum)

"""evaluation in terms of return values for the MOBMA"""

def eval_MORL_agent(agent, env, n_episodes, state_reset=None, random_init="no randomness", seed=None):
    agent.policy_dqn.eval()   
    reward_instr = 0
    reward_resc = 0
    reward_wait = 0
    in_water_sum = 0

    env.set_reward_type("MO")
    #env.set_render_mode("human")
    for _ in tqdm(range(n_episodes)):
        in_water = False
        state_history = []
        if state_reset:
            state, _ = env.reset(state=state_reset)
        else:
            state, _ = env.reset(random_init=random_init, seed=seed)
        terminated = False     
        truncated = False              

        # morally shielded policy
        while(not terminated and not truncated): 
            state_history.append(env.get_obs_dict())
            for person in env.persons:
                if person.status == bridge_person.Status.IN_WATER_AT_RISK:
                    in_water = True   
            # select morally permissible action  
            with torch.no_grad():
                #take an action that is conform with the agent's moral obligations according to its current reason theory
                action = agent.policy_dqn(agent.transformation(state)).argmax().item()

            # execute the action
            state,reward,terminated,truncated,_ = env.step(action)

            reward_instr += reward[0]
            reward_wait  += reward[1]
            reward_resc  += reward[2]

        state_history = []
                

        if in_water:
            in_water_sum += 1
            
    print("reward instr:", reward_instr, "reward push:", reward_wait, "reward resc:", reward_resc, "in_water_sum:", in_water_sum,)
