import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from Algorithm.DDPG.replay_buffer import Replay_buffer
from Algorithm.DDPG.network import Actor, Critic

class Noise():
    def __init__(self, action_dim,theta=0.15, mu=0, sigma=0.2):
        self.noise = np.ones(action_dim) * mu
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        
    def run(self, Type='normal'):
        if Type == 'normal':
            return np.random.normal(0,0.1,self.action_dim)
        elif Type == 'ou':
            self.noise += self.theta * (self.mu - self.noise) + self.sigma * np.random.randn(self.action_dim)
            return self.noise
            
class Agent():
    def __init__(self, env, args, device, 
                 gamma=0.99,
                 tau = 0.005,
                 actor_lr = 0.0001,
                 critic_lr = 0.001,   
                 train_start =200,
                 batch_size =100,
                 replay_buffer_size = 1000000
                 
                 ):
        #Hyperparameters
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_limit = env.action_space.high[0]  
        self.args = args
        self.device = device
        self.gamma = gamma
        self.tau =tau
        self.train_start =train_start
        self.batch_size = batch_size
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        
        self.replay_buffer_size =replay_buffer_size
        
        self.step = 0 #This is used for training interval
        
        #Network
        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        
        self.actor_target = Actor(self.state_dim, self.action_dim, self.action_limit).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        #Optimizer
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        #Noise type
        self.noise = Noise(self.action_dim)
        self.noise_type = 'ou'
       
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
                
                next_state, reward, done, _ = self.env.step(action)
            else:
                self.step +=1
                
                action = self.get_action(torch.Tensor(state).to(self.device))
                                
                next_state, reward, done, _ = self.env.step(action)
                
                self.RB.save_sample(state, action, reward, done, next_state)
                
                
                if self.step >= self.train_start:
                    self.train()
                    
            state = next_state
            total_reward += reward
            n_step += 1
                
            if done:
                break
        return n_step, float(total_reward)
    
    def get_action(self, state, evaluate=False):
    
        with torch.no_grad():
            action = self.actor(state)
        action = action.detach().cpu().numpy()
        
        if evaluate!=True:
            action += self.noise.run(self.noise_type)
            
        return np.clip(action, -self.action_limit, self.action_limit)
    
        

    
    def train(self):  

        #Sample a random minibatch of N transitions from Replay buffer
        states, actions, rewards, dones, next_states = self.RB.get_sample(self.batch_size)
        
        states = torch.FloatTensor(list(states)).view(-1,self.state_dim).to(self.device) # shape: [memory size, state_dim]
        actions = torch.FloatTensor(list(actions)).view(-1,self.action_dim).to(self.device) # shape: [memory size, action_dim]
        rewards = torch.FloatTensor(list(rewards)).view(-1,1).to(self.device) # shape: [memory size]
        dones = torch.FloatTensor(list(dones)).view(-1,1).to(self.device) # shape: [memory size, 1]
        next_states = torch.FloatTensor(list(next_states)).view(-1,self.state_dim).to(self.device) # shape: [memory size, state_dim]        
        
        #Set y_i
        next_action = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_action)
        y = rewards + self.gamma*next_Q*(1-dones)
        y = y.detach()
        
        Q = self.critic(states,actions)
        
        #Update the critic by minimizing the loss
        critic_loss= F.mse_loss(Q, y)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        #Update the actor policy using the sampled policy gradient
        actor_q = self.critic(states, self.actor(states))
        actor_loss = -actor_q.mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        #update the target networks
        for main_param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau*main_param.data + (1.0-self.tau)*target_param.data)
            
        for main_param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau*main_param.data + (1.0-self.tau)*target_param.data)
      