import torch

def visualize(agent, env, state_reset=None, random_init= "no randomness"):

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
                    action = agent.policy_dqn(agent.transformation(state)).argmax().item()

                state,_,terminated,truncated,_ = env.step(action)

