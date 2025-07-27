from torch import nn
import torch.nn.functional as F
import random
from collections import deque
import torch

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class DQN_2hiddenlayers(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.out = nn.Linear(h2_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x
    
class DQN_1hiddenlayer(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)  
        self.out = nn.Linear(h1_nodes, out_actions) 

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = self.out(x)         
        return x
    
class CNN_DQN(nn.Module):
    def __init__(self, input_channels, grid_height, grid_width, num_actions, k1_size=3, k2_size=3, stride=1, padding =1):
        super(CNN_DQN, self).__init__()

        out_channels = 64
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=k1_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=k2_size, stride=stride, padding=padding)

        # calculates flattened input size after convolutions (depends on grid size and kernel/stride)
        conv1_height, conv1_width = self.calculate_output_size(grid_height, grid_width, k1_size, stride, padding)
        conv2_height, conv2_width = self.calculate_output_size(conv1_height, conv1_width, k2_size, stride, padding)
        flattened_size = out_channels * conv2_height * conv2_width
        self.out = nn.Linear(in_features=flattened_size, out_features=num_actions)
        self.output_layer=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_size, out_features=num_actions)
        )
    def calculate_output_size(self, input_height, input_width, kernel_size, stride, padding):
            output_height = (input_height + 2 * padding - kernel_size) // stride + 1
            output_width = (input_width + 2 * padding - kernel_size) // stride + 1
            return output_height, output_width

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))  
        x = torch.flatten(x)
        x = self.out(x)   
        return x
    