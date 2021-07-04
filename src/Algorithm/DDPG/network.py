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
                nn.Linear(n_state, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, action_dim),
                )
        
        self.action_limit = action_limit


    def forward(self, x):         
        return self.action_limit*torch.tanh(self.layers(x))

            
class Critic(nn.Module):
    def __init__(self,n_state, action_dim):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(n_state+action_dim , 400),
                nn.ReLU(),
                nn.Linear(400 , 300),
                nn.ReLU(),
                nn.Linear(300,1),
                
                )

    def forward(self, state, action):
        x = torch.cat([state,action], dim=-1)
        return self.layers(x)
