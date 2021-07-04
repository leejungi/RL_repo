import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from Algorithm.PPO.replay_buffer import Replay_buffer
from Algorithm.PPO.network import Actor, Critic

class Agent():
    def __init__(self, env, args, device, 
                 gamma=0.99,
                 lam = 0.97,
                 actor_lr = 1e-5,
                 critic_lr = 2e-5,     
                 train_epoch=10,
                 batch_size =32,
                 replay_buffer_size = 1024
                 
                 ):
        #Hyperparameters
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_limit = env.action_space.high[0]  
        self.args = args
        self.device = device
        self.gamma = gamma
        self.lam =lam
        self.train_epoch =train_epoch
        self.batch_size = batch_size
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        
        self.replay_buffer_size =replay_buffer_size
        
        self.step = 0 #This is used for training interval
        
        #Network
        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)
        
        #Optimizer
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
       
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
                
                action, action_prob = self.get_action(torch.Tensor(state).to(self.device))
                                
                next_state, reward, done, _ = self.env.step(action)
                
                self.RB.save_sample(state, action, (reward+8.)/8., done, next_state, action_prob)
                
                
                if self.step == self.replay_buffer_size or done:
                    self.train()
                    self.RB.clear()
                    self.step = 0 
                    
            state = next_state
            total_reward += reward
            n_step += 1
                
            if done:
                break
        return n_step, float(total_reward)
    
    def get_action(self, state, evaluate=False):
        with torch.no_grad():
            mean, std = self.actor(state)
            dist = Normal(mean, std)
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = action.clamp(-2.0,2.0)
        
        if evaluate==True:
            return torch.tanh(mean).detach().cpu().numpy()
        return action.detach().cpu().numpy(), log_prob.item()

    
    def train(self):
        states, actions, rewards, dones, next_states, probs = self.RB.get_sample()

        states = torch.FloatTensor(list(states)).to(self.device) # shape: [memory size, state_dim]
        actions = torch.FloatTensor(list(actions)).view(-1,self.action_dim).to(self.device) # shape: [memory size, action_dim]
        rewards = torch.FloatTensor(list(rewards)).view(-1,1).to(self.device) # shape: [memory size]
        dones = torch.FloatTensor(list(dones)).view(-1,1).to(self.device) # shape: [memory size]
        next_states = torch.FloatTensor(list(next_states)).to(self.device) # shape: [memory size, state_dim]
        probs = torch.FloatTensor(list(probs)).view(-1,1).to(self.device) # shape: [memory size]
        probs = probs.detach()
        
        Return = []
        ret = 0
        pre_v = 0
        adv=0
        
        #GAE
        for t in reversed(range(len(rewards))):
            ret = rewards[t] + self.gamma*(1-dones[t])*ret
            Return.insert(0,ret)
            
        Return = torch.FloatTensor(Return).to(self.device).unsqueeze(1)
        
        Advantage = []
        pre_v = 0
        adv=0
        with torch.no_grad():
            V = self.critic(states).squeeze(1)
            
            for t in reversed(range(len(rewards))):            
                
                delta = rewards[t] + self.gamma*(1-dones[t])*pre_v - V[t]
                adv = delta + self.gamma*self.lam*(1-dones[t])*adv
                pre_v = V[t]
                Advantage.insert(0,adv)
            Advantage = np.array(Advantage,dtype=np.float32)
            Advantage = torch.FloatTensor(Advantage).to(self.device).unsqueeze(1)
            
        
       
        for _ in range(self.train_epoch):
            mean, std = self.actor(states)
            dist = Normal(mean,std)
            present_prob = dist.log_prob(actions)
            
            ratio = torch.exp(present_prob - probs)
            ratio1 = ratio*Advantage
            ratio2 = torch.clamp(ratio, 1.-0.2, 1.+0.2)*Advantage
            actor_loss = -torch.min(ratio1, ratio2).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optim.step()
          
        for _ in range(self.train_epoch):  
            value = self.critic(states)#.squeeze(1)
            critic_loss= torch.nn.functional.mse_loss(Return, value)
            
            
            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optim.step()