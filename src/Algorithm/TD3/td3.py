import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from Algorithm.TD3.replay_buffer import Replay_buffer
from Algorithm.TD3.network import Actor, Critic

            
class Agent():
    def __init__(self, env, args, device, 
                 gamma=0.99,
                 tau = 0.005,
                 noise=0.1,
                 target_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 actor_lr = 1e-3,
                 critic_lr = 1e-3,   
                 batch_size =100,
                 update_interval = 1,
                 gradient_step=1,
                 replay_buffer_size = 10000,
                 
                 
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
        self.noise = noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.gradient_step = gradient_step
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        
        
        
        self.replay_buffer_size =replay_buffer_size
        
        self.step = 0 #This is used for training interval
        
        #Network
        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit).to(self.device)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_limit).to(self.device)
        self.critic1 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2 = Critic(self.state_dim, self.action_dim).to(self.device)
        
        self.target_critic1 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic2 = Critic(self.state_dim, self.action_dim).to(self.device)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        #Optimizer
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)
        
       
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
                
            if self.step >= self.batch_size and test==False:
                for _ in range(self.gradient_step):
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
            action += self.noise * np.random.randn(self.action_dim)  
          
        return np.clip(action, -self.action_limit, self.action_limit)   
    
    def sync(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param*(1-self.tau) + param.data*self.tau)
            
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param*(1-self.tau) + param.data*self.tau)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param*(1-self.tau) + param.data*self.tau)     
        

    
    def train(self):  

        #Sample a random minibatch of N transitions from Replay buffer
        states, actions, rewards, dones, next_states = self.RB.get_sample(self.batch_size)
        
        states = torch.FloatTensor(list(states)).view(-1,self.state_dim).to(self.device) # shape: [memory size, state_dim]
        actions = torch.FloatTensor(list(actions)).view(-1,self.action_dim).to(self.device) # shape: [memory size, action_dim]
        rewards = torch.FloatTensor(list(rewards)).view(-1,1).to(self.device) # shape: [memory size]
        dones = torch.FloatTensor(list(dones)).view(-1,1).to(self.device) # shape: [memory size, 1]
        next_states = torch.FloatTensor(list(next_states)).view(-1,self.state_dim).to(self.device) # shape: [memory size, state_dim]        
     
        
        
        Q1 = self.critic1(states, actions)
        Q2 = self.critic2(states, actions)

        next_action = self.target_actor(next_states)
        epsilon = torch.normal(mean=0, std=self.target_noise, size=next_action.size()).to(self.device)
        epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip).to(self.device)
        next_action = torch.clamp(next_action+epsilon, -self.action_limit, self.action_limit).to(self.device)

        target_Q1 = self.target_critic1(next_states, next_action)
        target_Q2 = self.target_critic2(next_states, next_action)
        y = rewards + self.gamma*(1-dones)* torch.min(target_Q1, target_Q2)
      
        
        # Delayed policy update
        if self.step % self.policy_delay == 0:
           # Update policy network parameter
           actor_q = self.critic1(states, self.actor(states))
           actor_loss = -actor_q.mean()
           self.actor_optim.zero_grad()
           actor_loss.backward()
           self.actor_optim.step()
           
           self.sync()
      
        critic1_loss = F.mse_loss(Q1, y.detach())
        critic2_loss = F.mse_loss(Q2, y.detach())
        
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step() 
        
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        
    