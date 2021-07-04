import torch
import torch.nn as nn
import numpy as np
          
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self,n_state, action_dim, action_limit):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(n_state, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        
        self.action_limit = action_limit
        self.log_min =-2
        self.log_max =20

    def forward(self, x): 
        x=self.layers(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=self.log_min, max=self.log_max)
        return mean, log_std
                
        
        

            
class Q(nn.Module):
    def __init__(self,n_state, action_dim):
        super(Q, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(n_state+action_dim , 256),
                nn.ReLU(),
                nn.Linear(256 , 256),
                nn.ReLU(),
                nn.Linear(256,1),
                
                )

    def forward(self, state, action):
        x = torch.cat([state,action], dim=-1)
        return self.layers(x)
