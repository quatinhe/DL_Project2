import numpy as np
import torch
import torch.nn.functional as F

class EpsilonGreedyStrategy:
    def __init__(self, initial_epsilon=1.0, final_epsilon=0.01, decay_rate=0.995):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_rate = decay_rate
        self.epsilon = initial_epsilon
    
    def get_action(self, q_values, episode):
        # Linear decay
        self.epsilon = max(self.final_epsilon, 
                          self.initial_epsilon - episode * (self.initial_epsilon - self.final_epsilon) / 1000)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(-1, 2)  # Random action between -1 and 1
        else:
            return torch.argmax(q_values).item() - 1  # Best action
    
    def get_epsilon(self):
        return self.epsilon

class BoltzmannStrategy:
    def __init__(self, initial_temp=1.0, final_temp=0.1, decay_rate=0.995):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.decay_rate = decay_rate
        self.temperature = initial_temp
    
    def get_action(self, q_values, episode):
        # Exponential decay of temperature
        self.temperature = max(self.final_temp, 
                             self.initial_temp * (self.decay_rate ** episode))
        
        # Apply softmax to get action probabilities
        probs = F.softmax(q_values / self.temperature, dim=0)
        
        # Sample action from probability distribution
        action_idx = torch.multinomial(probs, 1).item()
        return action_idx - 1  # Convert to -1, 0, 1 range
    
    def get_temperature(self):
        return self.temperature 