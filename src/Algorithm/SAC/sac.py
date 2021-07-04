import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from Algorithm.SAC.replay_buffer import Replay_buffer
from Algorithm.SAC.network import Actor, Q, V

            
class Agent():
    def __init__(self, env, args, device, 
                 alpha =1,
                 gamma=0.99,
                 tau = 0.005,
                 actor_lr = 3e-4,
                 critic_lr = 3e-4,   
                 batch_size =256,
                 update_interval = 1,
                 gradient_step=1,
                 replay_buffer_size = 1000000
                 
                 ):
        #Hyperparameters
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_limit = env.action_space.high[0]  
        self.args = args
        self.device = device
        self.alpha =alpha
        self.gamma = gamma
        self.tau =tau
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.gradient_step = gradient_step
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        
        
        
        self.replay_buffer_size =replay_buffer_size
        
        self.step = 0 #This is used for training interval
        
        #Network
        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit).to(self.device)
        self.Q1 = Q(self.state_dim, self.action_dim).to(self.device)
        self.Q2 = Q(self.state_dim, self.action_dim).to(self.device)
        self.V = V(self.state_dim).to(self.device)
        self.target_V = V(self.state_dim).to(self.device)
        
        
        self.target_V.load_state_dict(self.V.state_dict())
        
        #Optimizer
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.Q1_optim = optim.Adam(self.Q1.parameters(), lr=self.critic_lr)
        self.Q2_optim = optim.Adam(self.Q2.parameters(), lr=self.critic_lr)
        self.V_optim = optim.Adam(self.V.parameters(), lr=self.critic_lr)
        
       
        #Replay buffer
        self.RB = Replay_buffer(self.replay_buffer_size)
        
    def run(self,test=False):
        
        total_reward = 0
        n_step = 0
        
        max_step = self.args.max_step
        
        state = self.env.reset()
        
        for s in range(max_step):
            
            
            if test:
                action  = self.get_action(torch.Tensor(state).to(self.device), evaluate=True)
                
                next_state, reward, done, _ = self.env.step(action* self.action_limit)
            else:
                self.step +=1
                
                action = self.get_action(torch.Tensor(state).to(self.device))
                                
                next_state, reward, done, _ = self.env.step(action* self.action_limit)
                
                self.RB.save_sample(state, action, reward, done, next_state)
                
                
           
                    
            state = next_state
            total_reward += reward
            n_step += 1
            if self.step >= self.batch_size and test==False:
                for _ in range(self.gradient_step):
                    self.train()    
                if self.step % self.update_interval ==0: 
                    self.sync()
            if done:
                break
        
        return n_step, float(total_reward)
    
    def get_action(self, state, evaluate=False):
    
        with torch.no_grad():
            mean, log_std = self.actor(state)
            std = torch.exp(log_std)
            dist = Normal(mean,std)
            action = torch.tanh(dist.sample()).detach().cpu().numpy()
        if evaluate ==True:
            return torch.tanh(mean)
        return action
    
    def sync(self):
        for target_param, param in zip(self.target_V.parameters(), self.V.parameters()):
            target_param.data.copy_(target_param*(1-self.tau) + param.data*self.tau)        
        

    
    def train(self):  

        #Sample a random minibatch of N transitions from Replay buffer
        states, actions, rewards, dones, next_states = self.RB.get_sample(self.batch_size)
        
        states = torch.FloatTensor(list(states)).view(-1,self.state_dim).to(self.device) # shape: [memory size, state_dim]
        actions = torch.FloatTensor(list(actions)).view(-1,self.action_dim).to(self.device) # shape: [memory size, action_dim]
        rewards = torch.FloatTensor(list(rewards)).view(-1,1).to(self.device) # shape: [memory size]
        dones = torch.FloatTensor(list(dones)).view(-1,1).to(self.device) # shape: [memory size, 1]
        next_states = torch.FloatTensor(list(next_states)).view(-1,self.state_dim).to(self.device) # shape: [memory size, state_dim]        
    
        #V update
        with torch.no_grad():
            mean, log_std = self.actor(states)
            std = log_std.exp()
            normal = Normal(mean, std)
            resample = normal.rsample()
            cur_action = torch.tanh(resample) 
            cur_log_pi = normal.log_prob(resample) - torch.log((1-cur_action.pow(2))+1e-9)
            cur_log_pi = cur_log_pi.sum(1, keepdim=True)
            
            next_Q1 = self.Q1(states, cur_action)
            next_Q2 = self.Q2(states, cur_action)

        #V(st+1) = Q(st,at) - alpha*logPi(at|st)
        next_V = torch.min(next_Q1, next_Q2) - self.alpha * cur_log_pi
            
        V = self.V(states)
        V_loss = F.mse_loss(V, next_V.detach())

        self.V_optim.zero_grad()
        V_loss.backward()
        self.V_optim.step()
        
        #Q update
        next_V = self.target_V(next_states)
        TD_target = rewards + self.gamma*(1-dones)*next_V
        
        Q1 = self.Q1(states, actions)
        Q2 = self.Q2(states, actions)
        
        Q1_loss = F.mse_loss(Q1, TD_target.detach())
        Q2_loss = F.mse_loss(Q2, TD_target.detach())
        
        self.Q1_optim.zero_grad()
        Q1_loss.backward()
        self.Q1_optim.step() 
        
        self.Q2_optim.zero_grad()
        Q2_loss.backward()
        self.Q2_optim.step()
        
        #Actor update
        mean, log_std = self.actor(states)
        std = log_std.exp()
        normal =Normal(mean,std)
        resample = normal.rsample()
        pi_action = torch.tanh(resample)
        log_pi = normal.log_prob(resample) - torch.log((1-pi_action.pow(2))+1e-9)
        log_pi = log_pi.sum(1, keepdim=True)
        
        Q1 = self.Q1(states, pi_action)
        Q2 = self.Q2(states, pi_action)
        
        actor_loss = (self.alpha*log_pi - torch.min(Q1,Q2)).mean()
        

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        