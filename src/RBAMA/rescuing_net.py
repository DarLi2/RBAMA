import numpy as np
import random
import torch
from torch import nn
import os
from src.utils.drl import DQN_2hiddenlayers, CNN_DQN, ReplayMemory
from src.RBAMA import RBAMA
from src.utils.plotting import plot_training_progress
from src.environments import registered_versions

def get_agent_info(agent_name):
    dir_name = RBAMA.get_dir_name()
    file_path = os.path.join(dir_name, agent_name + ".pth")
    agent_info = torch.load(file_path)
    return agent_info

def initilize_modules(agent, agent_info):
    agent.policy_dqn.load_state_dict(agent_info["policy_state_dict"])
    agent.target_dqn.load_state_dict(agent_info["target_state_dict"])

def setup_rescuing_agent(agent_name):
    agent_info = RBAMA.get_agent_info(agent_name)
    env = agent_info["env"]
    if agent_info["agent_type"] == "rescuing_agent":
        agent = Rescuing(env)
    elif agent_info["agent_type"] == "rescuing_agent_cnn":
        agent = RescuingCNN(env)
    initilize_modules(agent, agent_info)
    return (agent, env)

def save_agent(agent, agent_name):
    save_dict = {
        "policy_state_dict": agent.policy_dqn.state_dict(),
        "target_state_dict": agent.target_dqn.state_dict(),
        "env": agent.env,
        "agent_type": agent.agent_type
    }
    dir_name = RBAMA.get_dir_name()
    file_path= os.path.join(dir_name, agent_name + ".pth")

    torch.save(save_dict, file_path)

class Rescuing():
    def __init__(self, env, lr = 0.001, sync_rate = 1000, replay_memory_size=1000, mini_batch_size= 32):

        self.env = env
        self.agent_type = "rescuing_agent"

        #set up hyperparameters for DRL
        self.learning_rate_a = lr       
        self.discount_factor_g = 0.9          
        self.network_sync_rate = sync_rate        
        self.replay_memory_size = replay_memory_size    
        self.mini_batch_size = mini_batch_size        

        self.loss_fn = nn.MSELoss()          
        self.optimizer = None                

        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        #set up policy and target network
        self.policy_dqn = DQN_2hiddenlayers(in_states=self.num_states, h1_nodes=64, h2_nodes=64, out_actions=self.num_actions)
        self.target_dqn = DQN_2hiddenlayers(in_states=self.num_states, h1_nodes=64, h2_nodes=64, out_actions=self.num_actions)

    def transformation(self, observation):
        return  torch.tensor(observation, dtype=torch.float32)
    
    """
    trains the agent's policy 
    """
    def train(self, episodes, agent_name, env=None, random_init = "positions"):

        if not env:
            env = self.env
        
        epsilon = 1 # 0: no randomness; 1:completely random
        k = 5 #hyperparamter controlling the epsilon decay rate
        memory = ReplayMemory(self.replay_memory_size)

        # initially syn the policy and the target network (same parameters)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # set up the optimizer
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)

        # list to keep track of rewards collected per episode
        rewards_per_episode = np.zeros(episodes)

        # counter for keeping track of number of episodes where persons ended up in the water; i.e. counts the number of episodes, in which the reascuing network was activated and trained
        counter_resc_episodes = -1

        # count steps until policy and target network are synced
        step_count=0

        while counter_resc_episodes < episodes:
            episode_resc = False #a flag indicating if a person ended up in the water in the current episode
            state, _ = env.reset(random_init=random_init)  
            terminated = False      
            truncated = False       

            while(not terminated and not truncated):
                state = self.transformation(state)
                if "D" in env.get_lables():
                    if episode_resc == False:
                        counter_resc_episodes +=1
                        episode_resc = True
                    if random.random() >= epsilon:
                        with torch.no_grad():
                            action = self.policy_dqn(state).argmax().item()
                    else:
                        # select random action
                        action = env.action_space.sample()  # actions: 0=left,1=down,2=right,3=up, 4:save
                    # execute the agent's action choice
                    new_state,reward,terminated,truncated,_ = env.step(action)
                    new_state = self.transformation(new_state)
                    memory.append((state, action, new_state, reward, terminated)) 
                        
                else:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up, 4:save

                # execute the agent's action choice
                    new_state,reward,terminated,truncated,_ = env.step(action)
                    new_state = self.transformation(new_state)

                # update the state
                state = new_state

                # increment counter for synching the target network with the policy network
                step_count+=1

                if "D" in env.get_lables():
                    # update the policy network
                    if len(memory)>self.mini_batch_size: 
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch)   

                    # sync policy and target network
                    if step_count > self.network_sync_rate:
                        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                        step_count=0

            if episode_resc:
                rewards_per_episode[counter_resc_episodes-1] += reward

            epsilon=np.exp(-k * (counter_resc_episodes/episodes))


        env.close()
        sum_rewards = np.zeros(counter_resc_episodes)
        for x in range(counter_resc_episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-200):(x+1)])

        plot_training_progress(sum_rewards, agent_name)
            

    """
    uses a mini batch of experiences to update the agent's policy
    """
    def optimize(self, mini_batch):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * self.target_dqn(new_state).max()
                    )

            current_q = self.policy_dqn(state)
            current_q_list.append(current_q)

            
            target_q = self.target_dqn(state) 
            target_q[action] = target
            target_q_list.append(target_q)
                
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class RescuingCNN(Rescuing):
    def __init__(self, env, lr = 0.001, sync_rate = 1000, replay_memory_size=1000, mini_batch_size= 32, k1_size=3, k2_size=3, stride=1, padding =1):
        super().__init__(env, lr=lr, sync_rate=sync_rate, replay_memory_size=replay_memory_size, mini_batch_size=mini_batch_size)

        self.agent_type = "rescuing_agent_cnn"
        self.num_channels = self.env.observation_space.shape[0]  
        
        self.grid_height = env.bridge_map.height
        self.grid_width = env.bridge_map.width

        self.policy_dqn = CNN_DQN(
            input_channels=self.num_channels,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            num_actions=self.num_actions,
            k1_size=k1_size,
            k2_size=k2_size,
            stride=stride,
            padding=padding
        )
        self.target_dqn = CNN_DQN(
            input_channels=self.num_channels,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            num_actions=self.num_actions,
            k1_size=k1_size,
            k2_size=k2_size,
            stride=stride,
            padding=padding
        )
