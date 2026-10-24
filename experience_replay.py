import random
from collections import deque
import numpy as np
import torch


class ReplayMemory:
    def __init__(self, capacity, device):
        self.memory = deque(maxlen=capacity)
        self.device = device

    def cache(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, next_state, action, reward, done = zip(*batch)
        state      = torch.from_numpy(np.stack(state)).to(self.device)
        next_state = torch.from_numpy(np.stack(next_state)).to(self.device)
        action     = torch.tensor(action, dtype=torch.long, device=self.device)
        reward     = torch.tensor(reward, dtype=torch.float, device=self.device)
        done       = torch.tensor(done, dtype=torch.float, device=self.device)
        return state, next_state, action, reward, done

    def __len__(self):
        return len(self.memory)
