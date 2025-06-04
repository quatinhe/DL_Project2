import collections
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Add safety check to ensure we have enough samples
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer. Required: {batch_size}, Available: {len(self.buffer)}")
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states, axis=0),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0),
            np.array(dones, dtype=np.bool_)
        )

    def __len__(self):
        return len(self.buffer)
    
    def can_sample(self, batch_size):
        """Check if buffer has enough samples for the requested batch size"""
        return len(self.buffer) >= batch_size
