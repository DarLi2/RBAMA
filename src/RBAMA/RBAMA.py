import gymnasium as gym
import numpy as np
import random
import torch
from tqdm import tqdm
from torch import nn
from src.RBAMA import reasoning_unit
import os
from src.RBAMA import translator
from src.RBAMA import moral_judge
from src.environments import registered_versions
from src.utils import drl
from src.utils.drl import CNN_DQN
from src.environments.wrappers.multi_channel import Multi_Channel
from src.utils.plotting import plot_training_progress
from src.RBAMA import guard_net, rescuing_net
import logging

logger = logging.getLogger(__name__)

"""
methods for saving and loading an agent
"""
def get_agent_info(agent_name):
    dir_name = get_dir_name()
    file_path = os.path.join(dir_name, agent_name + ".pth")
    agent_info = torch.load(file_path)
    return agent_info

def load_modules(agent, agent_info):
    agent.policy_dqn.load_state_dict(agent_info["policy_state_dict"])
    agent.target_dqn.load_state_dict(agent_info["target_state_dict"])
    
    if "reasoning_unit_guard_state_dict" in agent_info and "reasoning_unit_rescuing_state_dict" in agent_info:
        guard_net_type = agent_info.get("guard_net_type", "Guard")
        rescuing_net_type = agent_info.get("rescuing_net_type", "Rescuing")
        
        if guard_net_type == "Bandit_Pushing":
            agent.reasoning_unit.guard_net = guard_net.Bandit_Pushing(agent.env)
        elif guard_net_type == "Guard_CNN":
            agent.reasoning_unit.guard_net = guard_net.Guard_CNN(agent.env)
        else:  
            agent.reasoning_unit.guard_net = guard_net.Guard(agent.env)
            
        if rescuing_net_type == "RescuingCNN":
            agent.reasoning_unit.rescuing_net = rescuing_net.RescuingCNN(agent.env)
        else:  
            agent.reasoning_unit.rescuing_net = rescuing_net.Rescuing(agent.env)
        
        agent.reasoning_unit.guard_net.policy_dqn.load_state_dict(agent_info["reasoning_unit_guard_state_dict"])
        agent.reasoning_unit.rescuing_net.policy_dqn.load_state_dict(agent_info["reasoning_unit_rescuing_state_dict"])

def setup_reasoning_agent(agent_name):
    agent_info = get_agent_info(agent_name)
    env = agent_info["env"]
    if agent_info["agent_type"] == "RBAMA":
        agent = RBAMA(env)
    elif agent_info["agent_type"] == "RBAMA_cnn":
        agent = RBAMA_CNN(env)
    load_modules(agent, agent_info)
    return (agent, env)

def get_dir_name():
    current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
    return os.path.join(current_file_dir, 'agents')

def save_agent(agent, agent_name):
    
    save_dict = {
        "policy_state_dict": agent.policy_dqn.state_dict(),
        "target_state_dict": agent.target_dqn.state_dict(),
        "env": agent.env,
        "agent_type": agent.agent_type,
        "reasoning_unit": agent.reasoning_unit
    }
    if getattr(agent.reasoning_unit, "guard_net", None):
        save_dict["reasoning_unit_guard_state_dict"] = agent.reasoning_unit.guard_net.policy_dqn.state_dict()
        guard_net_type = type(agent.reasoning_unit.guard_net).__name__
        save_dict["guard_net_type"] = guard_net_type
    if getattr(agent.reasoning_unit, "rescuing_net", None):
        save_dict["reasoning_unit_rescuing_state_dict"] = agent.reasoning_unit.rescuing_net.policy_dqn.state_dict()
        rescuing_net_type = type(agent.reasoning_unit.rescuing_net).__name__
        save_dict["rescuing_net_type"] = rescuing_net_type
    dir_name = get_dir_name()
    file_path= os.path.join(dir_name, agent_name + ".pth")
    logger.info(f"Agent saved to {file_path}")
    torch.save(save_dict, file_path)

"""implementation of the RBAMA (reason-based moral agent)"""
class RBAMA():
    def __init__(self, env, lr = 0.001, sync_rate = 1000, replay_memory_size=1000, mini_batch_size= 32):

        self.env = env
        self.agent_type = "RBAMA"

        #set up hyperparameters for DRL
        self.lr = lr        
        self.discount = 0.9            
        self.sr = sync_rate       
        self.replay_memory_size = replay_memory_size     
        self.mini_batch_size = mini_batch_size          

        self.loss_fn = nn.MSELoss()          
        self.optimizer = None                

        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        #set up policy and target network
        self.policy_dqn = drl.DQN_2hiddenlayers(in_states=self.num_states, h1_nodes=64, h2_nodes=64, out_actions=self.num_actions)
        self.target_dqn = drl.DQN_2hiddenlayers(in_states=self.num_states, h1_nodes=64, h2_nodes=64, out_actions=self.num_actions)
        
        #set tup the resaoning_unit 
        self.reasoning_unit = reasoning_unit.ReasoningUnit(env, self.num_actions)

        self.logger = logging.getLogger(__name__)


    """
    preparing the observation as input for the NN: flattens the elements of the observation dict and concatenates them  
    """
    def transformation(self, observation):
        return  torch.tensor(observation, dtype=torch.float32)

    """
    training loop for the instrumental policy
    """
    def train(self, episodes, agent_name,  env=None, random_init="positions", pb=True):

        if not env:
            env = self.env

        env.set_reward_type("instrumental")
        
        epsilon = 1 # 0: no randomness; 1:completely random
        k = 5 # decay rate for epsilon
        memory = drl.ReplayMemory(self.replay_memory_size)

        # initially sync the policy and the target network (same parameters)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # set up the optimizer
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.lr)

        rewards_per_episode = np.zeros(episodes)

        # count steps until policy and target network are synced
        step_count=0

        iterator = tqdm(range(episodes), desc="Training Instr") if pb else range(episodes)

        for i in iterator:
            state, info = env.reset(random_init=random_init) 
            terminated = False      
            truncated = False      
            steps_in_episode = 0

            while(not terminated and not truncated):

                # select action epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up, 4:save, 5:idle
                else:
                    # select best action            
                    with torch.no_grad():
                        action = self.policy_dqn(self.transformation(state)).argmax().item()

                # execute the agent's action choice
                new_state,reward,terminated,truncated,_ = env.step(action)

                if reward != 0:
                    pass

                # add sample to memory
                memory.append((state, action, new_state, reward, terminated)) 

                # update the state
                state = new_state

                # increment counter for synching the target network with the policy network
                step_count+=1

                # update the policy network
                if len(memory)>self.mini_batch_size: 
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch) 
                
                # sync policy and target network
                if step_count > self.sr:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    step_count=0

            if reward != 0:
                rewards_per_episode[i] = reward

            # reduce chance to randomly pick an action
            epsilon=np.exp(-k * (i/episodes))

        env.close()

        #sum_rewards = np.zeros(episodes)
        sum_rewards = np.array([np.sum(rewards_per_episode[max(0, x-200):x+1]) for x in range(episodes)])

        plot_training_progress(sum_rewards, agent_name)

    """
    training loop for the instrumental policy (shielded)
    """
    def train_shielded(self, episodes, agent_name, obl, env=None, random_init="positions", pb=True):

        if not env:
            env = self.env
        
        epsilon = 1 # 0: no randomness; 1:completely random
        k=5 # decay rate for epsilon
        memory = drl.ReplayMemory(self.replay_memory_size)

        # initially syn the policy and the target network (same parameters)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # set up the optimizer
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.lr)

        rewards_per_episode = np.zeros(episodes)

        # count steps until policy and target network are synced
        step_count=0

        iterator = tqdm(range(episodes), desc="Training Instr") if pb else range(episodes)

        for i in iterator:
            state, _ = env.reset(random_init=random_init)  
            terminated = False      
            truncated = False       

            while(not terminated and not truncated):

                morally_permissible_actions = self.reasoning_unit.primitive_actions("C", state)
                state_transformed = self.transformation(state)
                if random.random() < epsilon:
                # select random action
                    action = env.action_space.sample()
                    while action not in morally_permissible_actions:
                        action = env.action_space.sample()
                else: 
                    with torch.no_grad():
                        action_preferences = [tensor.item() for tensor in self.policy_dqn(state_transformed)]
                    action = action_preferences.index(max(action_preferences))
                    choice_list = list(action_preferences)
                    while action not in morally_permissible_actions:
                        choice_list.remove(max(choice_list))
                        action = action_preferences.index(max(choice_list))

                # execute the agent's action choice
                new_state,reward,terminated,truncated,_ = env.step(action)

                # add sample to memory
                memory.append((state, action, new_state, reward, terminated)) 

                # update the state
                state = new_state

                # increment counter for synching the target network with the policy network
                step_count+=1

                # update the policy network
                if len(memory)>self.mini_batch_size: 
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch) 
                
                if step_count > self.sr:
                    self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                    step_count=0

            rewards_per_episode[i] += reward

            # reduce chance to randomly pick an action
            epsilon=np.exp(-k * (i/episodes))

        env.close()

        sum_rewards = np.array([np.sum(rewards_per_episode[max(0, x-200):x+1]) for x in range(episodes)])

        plot_training_progress(sum_rewards, agent_name)
    

    """
    use a mini batch of experiences to update the agent's policy
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
                        reward + self.discount * self.target_dqn(self.transformation(new_state)).max()
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

    """
    train the agent's reasoning
    """
    def train_reasons(self, episodes, env, judge=moral_judge.JudgePrioR(translator=translator.Translator()), random_init="positions"):

        if not env:
            env = self.env

        env.metadata["render_fps"] = 2

        self.policy_dqn.eval()  

        for _ in range(episodes):
            state, _ = env.reset(random_init=random_init) 
            terminated = False      
            truncated = False      
            judgement = None       

            while(not terminated and not truncated): 
                #if the agent receives feedback from the moral judge, it updates its reason theory
                if judgement:
                    self.reasoning_unit.update(judgement)
                morally_permissible_actions = self.filter_actions(env, state)
                # if there is only one morally permissible action, bypass the NN that optimizes for achieving the instrumental policy and execute the action directly
                if len(morally_permissible_actions) == 1:
                    action = morally_permissible_actions[0]
                # otherwise, select the action that is morally conform with all duties and has the highest expecet instrumental value
                else:
                    with torch.no_grad():
                        action_preferences = [tensor.item() for tensor in self.policy_dqn(self.transformation(state))]
                    action = action_preferences.index(max(action_preferences))
                    choice_list = list(action_preferences)
                    while action not in morally_permissible_actions:
                        choice_list.remove(max(choice_list))
                        action = action_preferences.index(max(choice_list))

                # judge gives feedback if the chosen action was morally permissible
                judgement = judge.judgement(env, action)
                # execute the action
                state,_,terminated,truncated,_ = env.step(action)
        
        env.close()


    def filter_actions(self, env, state):
        #get the action types toward whose realization the agent has an all things considered normative reason (the agent's moral obligations)
        lables = env.get_lables()
        moral_obligations = self.reasoning_unit.moral_obligations(lables, state)
        morally_permissible_actions = self.reasoning_unit.primitive_actions(moral_obligations, state)
  
        return morally_permissible_actions

"""reasoning agent integrating a CNN for learning the instrumental policy"""
class RBAMA_CNN(RBAMA):
    def __init__(self, env):
        super().__init__(env)
        self.env=Multi_Channel(env)
        self.agent_type = "RBAMA_cnn"

        self.num_channels = self.env.observation_space.shape[0]  
        self.grid_height = env.bridge_map.height
        self.grid_width = env.bridge_map.width

        # set up of CNNs
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
