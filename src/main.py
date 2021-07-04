import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from collections import deque
# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--env', type=str, default='Pendulum-v0', 
                    help='pendulum environment')
parser.add_argument('--algo', type=str, default='td3', 
                    help='Select Algorithm in TRPO, DDPG, PPO, SAC, SAC with autotuning, TD3')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--episode', type=int, default=50000, 
                    help='Iterations for train')
parser.add_argument('--test_interval', type=int, default=500, 
                    help='Test interval while training')
parser.add_argument('--log_interval', type=int, default=10, 
                    help='Train log interval while training')
parser.add_argument('--max_step', type=int, default=200,
                    help='max episode step')
parser.add_argument('--tensorboard', action='store_false', default=True)
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

if args.device == 'cuda':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')
    
if args.algo == 'trpo':
    from Algorithm.TRPO.trpo import Agent
elif args.algo =='ddpg':
    from Algorithm.DDPG.ddpg import Agent
elif args.algo =='ppo':
    from Algorithm.PPO.ppo import Agent
elif args.algo =='sac':
    from Algorithm.SAC.sac import Agent
elif args.algo =='asac':
    from Algorithm.ASAC.asac import Agent
elif args.algo =='td3':
    from Algorithm.TD3.td3 import Agent
    
def main():
    # Create a SummaryWriter object by TensorBoard
    if args.tensorboard:
        dir_name = 'tf_board/' + args.env + '/' \
                           + args.algo \
                           + '_s_' + str(args.seed) \
                           + '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        writer = SummaryWriter(log_dir=dir_name)
        
    #Environment Setup
    env = gym.make(args.env)
    
    # Set a random seed
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':        
        torch.cuda.manual_seed_all(args.seed)
        
    #Algorithm Agent
    agent = Agent(env, args, device)
    
    start_time = time.time()
    total_return = deque(maxlen=100)
    eps = 1e-9
    

    for i in range(args.episode):
        
        n_step, total_reward = agent.run()
        Return = total_reward/(n_step+eps)
        total_return.append(Return)
        
        avg_return = np.mean(total_return)
        

        if args.tensorboard:
            # writer.add_scalar('Train/AverageReturns', avg_return, i)
            # writer.add_scalar('Train/EpisodeReturns', total_reward, i)
            writer.add_scalar('Train/EpisodeReturns', Return, i) #returns per episode
            writer.add_scalar('Train/AverageReturns', avg_return, i) #average returns during episode

        if (i+1)%args.log_interval ==0:
            print('Episodes:', i + 1)
            # print('EpisodeReturn:', round(total_reward, 2))
            # print('100AverageReturn:', round(avg_return, 2))
            print('\tEpisodeReturn:', round(total_reward, 2))
            print('\tAverageReturn:', round(avg_return, 2))
            print('\tTime:', int(time.time() - start_time))
        
        if (i+1)%args.test_interval ==0:
            test_return = []
            eval_sum_returns = 0.
            eval_num_episodes = 0
            for _ in range(10):
                n_step, total_reward = agent.run(test=True)
                eval_sum_returns += total_reward
                eval_num_episodes += 1
                test_return.append(total_reward /(n_step + eps))
            avg_return = np.mean(test_return)
            # avg_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0

            if args.tensorboard:
                writer.add_scalar('Test/Test_AverageReturn', avg_return, i)
                writer.add_scalar('Test/EpisodeReturns', total_reward, i)
            print('Test Episodes:', i + 1)
            print('\tTest_AverageReturn:', round(avg_return, 2))
            print('\tTime:', int(time.time() - start_time))
        
if __name__ == "__main__":
    main()
