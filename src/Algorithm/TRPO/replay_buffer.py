import numpy as np
import torch
from collections import deque

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
    
    def get_sample(self):
        return self.states, self.actions, self.rewards, self.dones, self.next_states
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_states.clear()
    
    def __len__(self):
        return len(self.buffer)
    
    