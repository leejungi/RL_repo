import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from Algorithm.TRPO.replay_buffer import Replay_buffer
from Algorithm.TRPO.network import Actor, Critic

class Agent():
    def __init__(self, env, args, device, 
                 gamma=0.99,
                 lam = 0.97,
                 critic_lr = 1e-3,
                 delta = 0.01,
                 backtrack_iter=10,
                 backtrack_coeff=1.0,
                 backtrack_alpha=0.5,
                 replay_buffer_size = 2048
                 
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
        self.critic_lr = critic_lr
        self.delta = delta
        self.backtrack_iter = backtrack_iter
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_alpha = backtrack_alpha
        self.replay_buffer_size =replay_buffer_size
        
        self.step = 0 #This is used for training interval
        
        #Network
        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit).to(self.device)
        self.old_actor = Actor(self.state_dim, self.action_dim, self.action_limit).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)
        
        #Optimizer
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
                action, _, _, _ = self.actor(torch.Tensor(state).to(self.device))
                action = action.detach().cpu().numpy() 
                
                next_state, reward, done, _ = self.env.step(action)
            else:
                self.step +=1
                
                _, _, action, _ = self.actor(torch.Tensor(state).to(self.device))
                
                action = action.detach().cpu().numpy()
                
                next_state, reward, done, _ = self.env.step(action)
                
                self.RB.save_sample(state, action, reward, done, next_state)
                
                if self.step == self.replay_buffer_size:
                    self.train()
                    self.step = 0 
                    
            state = next_state
            total_reward += reward
            n_step += 1
                
            if done:
                break
        return n_step, float(total_reward)
    
    
    def train(self):
        states, actions, rewards, dones, _ = self.RB.get_sample()

        states = torch.FloatTensor(list(states)).to(self.device) # shape: [memory size, state_dim]
        actions = torch.FloatTensor(list(actions)).to(self.device) # shape: [memory size, action_dim]
        rewards = torch.FloatTensor(list(rewards)).to(self.device) # shape: [memory size]
        dones = torch.FloatTensor(list(dones)).to(self.device) # shape: [memory size]

        Return = []
        Advantage = []
        ret = 0
        pre_v = 0
        adv=0
        
        V = self.critic(states).squeeze(1)
        
        #GAE
        for t in reversed(range(self.replay_buffer_size)):
            ret = rewards[t] + self.gamma*(1-dones[t])*ret
            Return.insert(0,ret)
            
            delta = rewards[t] + self.gamma*(1-dones[t])*pre_v - V[t]
            adv = delta + self.gamma*self.lam*(1-dones[t])*adv
            pre_v = V[t]
            Advantage.insert(0,adv)
        
        
        Return = torch.FloatTensor(Return).to(self.device).detach()
        
        Advantage = np.array(Advantage,dtype=np.float32)
        Advantage = (Advantage-Advantage.mean())/Advantage.std()
        Advantage = torch.FloatTensor(Advantage).to(self.device)
        
        #Critic Update
        for _ in range(80):
            V = self.critic(states).squeeze(1)
            critic_loss = F.mse_loss(V,Return)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            
        #Actor update
        _, _, _, log_pi_old = self.actor(states, actions)
        log_pi_old = log_pi_old.detach()
        _, _, _, log_pi = self.actor(states, actions)
        
        ratio = torch.exp(log_pi - log_pi_old)
        actor_loss_old = (ratio*Advantage).mean()
        
        #Compute gradient
        gradient = torch.autograd.grad(actor_loss_old, self.actor.parameters())
        gradient = self.flat_grad(gradient)
        
        #Conjugate Gradient
        search_dir = self.cg(states, gradient.data)
        gHg = (self.fisher_vector_product(states, search_dir)*search_dir).sum(0)
        step_size = torch.sqrt(2*self.delta/gHg)
        old_params = self.flat_params(self.actor)
        self.update_model(self.old_actor,old_params)
        
        expected_improve = (gradient * step_size * search_dir).sum(0, keepdim=True)
        for i in range(self.backtrack_iter):
            # Backtracking line search            
            params = old_params + self.backtrack_coeff * step_size * search_dir
            self.update_model(self.actor, params)
             
            _, _, _, log_pi = self.actor(states, actions)
            ratio = torch.exp(log_pi - log_pi_old)
            actor_loss = (ratio*Advantage).mean()
             
            loss_improve = actor_loss - actor_loss_old
            expected_improve *= self.backtrack_coeff
            improve_condition = loss_improve / expected_improve
             
            kl = self.kl_divergence(new_actor=self.actor, old_actor=self.old_actor, states=states)

            if kl < self.delta and improve_condition > self.backtrack_alpha:
                break
             
            if i == self.backtrack_iter-1:
                params = self.flat_params(self.old_actor)
                self.update_model(self.actor, params)
             
            self.backtrack_coeff *= 0.5
        

            
        
    def flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten
    
    def flat_grad(self, grads):
        grad_flatten = []
        for grad in grads:
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten
    
    def cg(self, states, b, cg_iters=10, EPS=1e-8, residual_tol=1e-10):
        # Conjugate gradient algorithm
        x = torch.zeros(b.size()).to(self.device)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r,r).to(self.device)
            
        for _ in range(cg_iters):
            Ap = self.fisher_vector_product(states, p)
            alpha = rdotr / (torch.dot(p, Ap).to(self.device) + EPS)
           
            x += alpha * p
            r -= alpha * Ap
           
            new_rdotr = torch.dot(r, r)
            p = r + (new_rdotr / rdotr) * p
            rdotr = new_rdotr
          
            if rdotr < residual_tol:
               break
        return x
    
    def kl_divergence(self, new_actor, old_actor, states):
        mu, std, _,  _ = new_actor(torch.Tensor(states))
        mu_old, std_old, _, _  = old_actor(torch.Tensor(states))
        
        mu_old = mu_old.detach()
        std_old = std_old.detach()
    
        # kl divergence between old policy and new policy : D( pi_old || pi_new )
        kl = torch.log(std/std_old) + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
             (2.0 * std.pow(2)) - 0.5
        return kl.sum(-1, keepdim=True).mean()
    
    def flat_hessian(self,hessians):
        hessians_flatten = []
        for hessian in hessians:
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten
    
    def fisher_vector_product(self, states, p):
        p.detach()
        kl = self.kl_divergence(new_actor=self.actor, old_actor=self.actor, states=states)
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad = self.flat_grad(kl_grad)  # check kl_grad == 0
    
        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, self.actor.parameters())
        kl_hessian_p = self.flat_hessian(kl_hessian_p)
    
        return kl_hessian_p + 0.1 * p
    
    def update_model(self, model, new_params):
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length