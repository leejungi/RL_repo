import random
from collections import deque
import numpy as np
class Replay_buffer:
    def __init__(self, size):
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.dones = deque(maxlen=size)
        self.next_states = deque(maxlen=size)

    def save_sample(self, state, action, reward, done, next_state):
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
    
    def get_sample(self, batch_size):
        ind = np.random.randint(0, len(self.states), size=batch_size)
        states, actions, rewards, dones, next_states = [], [], [], [], []
        
        for i in ind:
            states.append(self.states[i])
            actions.append(self.actions[i])
            rewards.append(self.rewards[i])
            dones.append(self.dones[i])
            next_states.append(self.next_states[i])
        return states, actions, rewards, dones, next_states
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_states.clear()
    
    def __len__(self):
        return len(self.states)
    
    