import torch
import torch.nn as nn
import torch.nn.init
from torch.distributions import Categorical, Normal
import numpy as np

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
        
class Actor(nn.Module):
    def __init__(self,state_dim, action_dim, action_limit):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(state_dim, 64),
                # nn.ReLU(),
                nn.Tanh(),
                nn.Linear(64, 64),
                # nn.ReLU(),
                nn.Tanh(),
                )
        self.mean= nn.Linear(64, action_dim)
        self.log_std = np.ones(action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.Tensor(self.log_std))
        # self.log_std= nn.Linear(64, action_dim)
        self.action_limit = action_limit
        # self.apply(weights_init_)

        self.log_min = -2
        self.log_max = 20

    def forward(self, x, pi=None):         
        x = self.layers(x)
        mean = self.mean(x)
        # log_std = self.log_std(x)
        # log_std = torch.clamp(log_std, min=self.log_min, max = self.log_max)
        log_std = self.log_std

        std = torch.exp(log_std)
        dist = Normal(mean, std)
        if pi == None:
            pi = dist.sample()  
        # action = torch.tanh(pi.float()).detach().cpu().numpy()
        log_action = dist.log_prob(pi).sum(dim=-1)
        
        # action = action.detach().cpu().numpy()
        
        return mean*self.action_limit, std, pi*self.action_limit, log_action
            
            
class Critic(nn.Module):
    def __init__(self,state_dim):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                )
        # self.apply(weights_init_)

    def forward(self, state):         
        return self.layers(state)
