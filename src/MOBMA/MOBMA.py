import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from tqdm import tqdm
from torch import nn
import os
from collections import OrderedDict
from src.environments import registered_versions
from src.MOBMA.ethical_momdp_wrapper import Ethical_MOMDP_Wrapper
from src.environments.wrappers.multi_channel import Multi_Channel
from src.utils.drl import CNN_DQN
from src.utils import drl
import torch
import torch.nn as nn

def get_dir_name():
    current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
    return os.path.join(current_file_dir, 'agents')

def get_agent_info(agent_name):
    dir_name = get_dir_name()
    file_path = os.path.join(dir_name, agent_name + ".pth")
    agent_info = torch.load(file_path)
    return agent_info

def setup(agent_name):
    agent_info = get_agent_info(agent_name)
    env = agent_info["env"]
    e_w =  agent_info["e_w"]
    if agent_info["agent_type"] == "drl_agent":
        agent = DRL_agent(env)
    elif agent_info["agent_type"] == "drl_cnn":
        agent = DRL_CNN(env)
    agent.policy_dqn = agent_info["state_dict"]
    agent.target_dqn = agent_info["state_dict"]
    return (agent, env)

def save_agent(agent, agent_name):
    save_dict = {
        "state_dict": agent.policy_dqn,
        "e_w": agent.e_w,
        "env": agent.env,
        "agent_type": agent.agent_type,
    }
    dir_name = get_dir_name()
    file_path= os.path.join(dir_name, agent_name + ".pth")

    torch.save(save_dict, file_path)

class DRL_agent():
    def __init__(self, env, e_w = 1):

        self.agent_type = "drl_agent"

        #setup the environment
        self.env = env
        self.e_w = e_w
        self.env = Ethical_MOMDP_Wrapper(env, e_w=e_w, obs_config='scalarized')

        #setup training parameters
        self.learning_rate_a = 0.001         
        self.discount_factor_g = 0.9         
        self.network_sync_rate = 1000          
        self.replay_memory_size = 1000       
        self.mini_batch_size = 32              

        #setup the NN 
        self.loss_fn = nn.MSELoss()          
        self.optimizer = None                
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        #set up policy and target network
        self.policy_dqn = drl.DQN_2hiddenlayers(in_states=self.num_states, h1_nodes=64, h2_nodes=64, out_actions=self.num_actions)
        self.target_dqn = drl.DQN_2hiddenlayers(in_states=self.num_states, h1_nodes=64, h2_nodes=64, out_actions=self.num_actions)

    def transformation(self, observation: OrderedDict) -> torch.Tensor:
        return torch.tensor(observation, dtype=torch.float32)

    def train(self, episodes, agent_name, random_init="positions", env=None):

        #set up the environemt
        if not env:
            env = self.env
        
        epsilon = 1 # 0: no randomness; 1:completely random
        memory = drl.ReplayMemory(self.replay_memory_size)

        # initially syn the policy and the target network (same parameters)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # set up the optimizer
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = np.zeros(episodes)

        # count steps until policy and target network are synced
        step_count=0

        with tqdm(range(episodes)) as pbar:
            
            for i in pbar:
                state, _ = env.reset(random_init=random_init)  
                terminated = False     
                truncated = False        

                discounted_return = 0
                cum_discount = 1

                while(not terminated and not truncated):

                    # select action based on epsilon-greedy
                    if random.random() < epsilon:
                        # select random action
                        action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                    else:
                        # select best action            
                        with torch.no_grad():
                            action = self.policy_dqn(self.transformation(state)).argmax().item()

                    # execute the agent's action choice
                    new_state,reward,terminated,truncated,_ = env.step(action)

                    # add sample to memory
                    memory.append((state, action, new_state, reward, terminated)) 

                    # update the state
                    state = new_state

                    # increment counter for synching the target network with the policy network
                    step_count+=1

                    discounted_return += cum_discount * reward
                    cum_discount *= self.discount_factor_g
                

                # rewards collected per episode
                if reward != 0:
                    rewards_per_episode[i] = reward

                # update the policy network
                if len(memory)>self.mini_batch_size: 
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch)        

                # sync policy and target network
                if step_count > self.network_sync_rate:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    step_count=0

                k = 5
                epsilon=np.exp(-k * (i/episodes))


        env.close()

        #create a plot for the training process (average rewward per episode)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])

        plt.figure()
        plt.plot(sum_rewards)
        plt.title('Training Process')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of Rewards (last 100 episodes)')
        
        #save the plot
        save_folder = 'saved_plots'  
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if agent_name:
            save_path = os.path.join(save_folder, agent_name + '.png')
        else:
            save_path = os.path.join(save_folder, self.env_name + '.png')
        plt.savefig(save_path)

    # update policy 
    def optimize(self, mini_batch):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * self.target_dqn(self.transformation(new_state)).max()
                    )

            current_q = self.policy_dqn(self.transformation(state))
            current_q_list.append(current_q)

            target_q = self.target_dqn(self.transformation(state)) 
            target_q[action] = target
            target_q_list.append(target_q)
                
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class DRL_CNN(DRL_agent):
    def __init__(self, env, e_w=1):
        super().__init__(env , e_w=e_w)
        self.env=Multi_Channel(env)

        self.agent_type = "drl_cnn"

        self.num_channels = self.env.observation_space.shape[0]  
        self.grid_height = env.bridge_map.height
        self.grid_width = env.bridge_map.width

        self.policy_dqn = CNN_DQN(
            input_channels=self.num_channels,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            num_actions=self.num_actions
        )
        self.target_dqn = CNN_DQN(
            input_channels=self.num_channels,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            num_actions=self.num_actions
        )



