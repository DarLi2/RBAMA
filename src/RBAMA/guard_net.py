from src.environments import registered_versions
import numpy as np
import random
import torch
from tqdm import tqdm
from torch import nn
import os
from src.utils import drl
from src.RBAMA import RBAMA
from src.utils.drl import DQN_2hiddenlayers, CNN_DQN, ReplayMemory


def get_agent_info(agent_name):
    dir_name = RBAMA.get_dir_name()
    file_path = os.path.join(dir_name, agent_name + ".pth")
    agent_info = torch.load(file_path)
    return agent_info

def initilize_modules(agent, agent_info):
    agent.policy_dqn.load_state_dict(agent_info["policy_state_dict"])
    if agent.target_dqn is not None:  
        agent.target_dqn.load_state_dict(agent_info["target_state_dict"])

def setup_on_bridge(agent_name):
    agent_info = RBAMA.get_agent_info(agent_name)
    env = agent_info["env"]
    if agent_info["agent_type"] == "guard":
        agent = Guard(env)
    elif agent_info["agent_type"] == "guard_cnn":
        agent = Guard_CNN(env)
    elif agent_info["agent_type"] == "bandit_pushing":
        agent = Bandit_Pushing(env)
    initilize_modules(agent, agent_info)
    return (agent, env)

def save_agent(agent, agent_name):
    save_dict = {
        "policy_state_dict": agent.policy_dqn.state_dict(),
        "env": agent.env,
        "agent_type": agent.agent_type
    }
    
    # Only save target network state dict if it exists (Bandit_Pushing doesn't have one)
    if hasattr(agent, 'target_dqn') and agent.target_dqn is not None:
        save_dict["target_state_dict"] = agent.target_dqn.state_dict()
    
    dir_name = RBAMA.get_dir_name()
    file_path= os.path.join(dir_name, agent_name + ".pth")

    torch.save(save_dict, file_path)

class Guard():
    def __init__(self, env, lr = 0.001, sync_rate = 1000, replay_memory_size=1000, mini_batch_size= 32):

        self.env = env
        self.agent_type = "guard_agent"

        #set up hyperparameters for DRL
        self.learning_rate_a = lr         
        self.discount_factor_g = 0.9 
        self.network_sync_rate = sync_rate         
        self.replay_memory_size = replay_memory_size      
        self.mini_batch_size = mini_batch_size         

        self.loss_fn = nn.MSELoss()                   

        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        #set up policy and target network
        self.policy_dqn = drl.DQN_2hiddenlayers(in_states=self.num_states, h1_nodes=64, h2_nodes=64, out_actions=self.num_actions)
        self.target_dqn = drl.DQN_2hiddenlayers(in_states=self.num_states, h1_nodes=64, h2_nodes=64, out_actions=self.num_actions)

        self.optimizer = None  


    """
    preparing the observation as input for the NN: flattens the elements of the observation dict and concatenates them  
    """
    def transformation(self, observation):
#
        return  torch.tensor(observation, dtype=torch.float32)
    
    """
    trains the agent's policy 
    """
    def train(self, episodes, agent_name, env=None, pb =True):

        if not env:
            env = self.env
        env.set_reward_type("constraint")
        
        epsilon = 1 # 0: no randomness; 1:completely random
        k = 5 #hyperparameter controlling the epsilon decay rate
        memory = drl.ReplayMemory(self.replay_memory_size)

        # initially sync the policy and the target network (same parameters)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # count steps until policy and target network are synced
        step_count=0

        iterator = tqdm(range(episodes), desc="Training Guard") if pb else range(episodes)
            
        for i in iterator:
            state, _ = env.reset(random_init = "positions")  
            terminated = False      
            truncated = False         

            while(not terminated and not truncated):
                state = self.transformation(state)

                # only activate the network (and use the sample for the training process) if there is a person standing on the bridge
                if "B" in env.get_lables():
                    if random.random() >= epsilon:
                        with torch.no_grad():
                            action = self.policy_dqn(state).argmax().item()
                    else:
                        action = env.action_space.sample()
                    new_state,reward,terminated,truncated,_ = env.step(action)
                    new_state = self.transformation(new_state)
                    # add sample to memory
                    memory.append((state, action, new_state, reward, terminated)) 
                else:
                    # select random action
                    action = env.action_space.sample()  # actions: 0=left,1=down,2=right,3=up, 4:save
                    # execute the agent's action choice
                    new_state,reward,terminated,truncated,_ = env.step(action)
                    new_state = self.transformation(new_state)

                # update the state
                state = new_state

                # increment counter for synching the target network with the policy network
                step_count+=1

                # update the policy network
            if len(memory)>self.mini_batch_size: 
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch) 

            # sync policy and target network
            if step_count > self.network_sync_rate:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                step_count=0   

            # reduce change to randomly pick an action
            epsilon=np.exp(-k * (i/episodes))

        env.close()


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

class Guard_CNN(Guard):
    def __init__(self, env):
        super().__init__(env)

        self.agent_type = "guard_cnn"
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

    def transformation(self, observation):
        return torch.as_tensor(observation, dtype=torch.float32)

class Bandit_Pushing(Guard_CNN):
    def __init__(self, env):
        super().__init__(env)
        
        # adjust agent type
        self.agent_type = "bandit_pushing"
        
        # remove target network 
        self.target_dqn = None  

        # set optimizer
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)
        
    def train(self, episodes, agent_name, env=None, pb=True):
        if not env:
            env = self.env
        
        env.set_reward_type("waiting")
        
        epsilon = 1  
        k = 5 #hyperparameter controlling the epsilon decay rate
        memory = drl.ReplayMemory(self.replay_memory_size)
        
        iterator = tqdm(range(episodes), desc="Training Guard") if pb else range(episodes)
        
        for i in iterator:
            state, _ = env.reset(random_init="positions")
            state = self.transformation(state)
            
            if "B" in env.get_lables():
            # select action based on epsilon-greedy
                if random.random() >= epsilon:
                    with torch.no_grad():
                        action = self.policy_dqn(state).argmax().item()
                else:
                    action = env.action_space.sample()
                _, reward, _, _, _ = env.step(action)
                # add sample to memory
                memory.append((state, action, -reward))
            
            
            # update policy using mini-batch
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch)
            
            epsilon = np.exp(-k * (i / episodes))
        
        env.close()
    
    def optimize(self, mini_batch):
        current_q_list = []
        target_q_list = []
        
        for state, action, reward in mini_batch:
            target = torch.FloatTensor([reward])  # no future reward considerations
            
            current_q = self.policy_dqn(self.transformation(state))
            current_q_list.append(current_q)
            
            target_q = current_q.clone()
            target_q[action] = target
            target_q_list.append(target_q)
            
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
