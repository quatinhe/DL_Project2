import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from snake_game import SnakeGame
from snake_visualizer import SnakeVisualizer
import time
import pygame
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from replay_buffer import ReplayBuffer
from exploration_strategies import EpsilonGreedyStrategy, BoltzmannStrategy

# Set device for GPU acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class DQN(nn.Module):
    def __init__(self, input_channels=3, num_actions=3):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # Use reshape with automatic size calculation
        x = x.reshape(x.size(0), -1)
        # Update fc1 to handle dynamic input size
        if not hasattr(self, '_fc1_input_size'):
            self._fc1_input_size = x.size(1)
            self.fc1 = nn.Linear(self._fc1_input_size, 512).to(x.device)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, input_channels=3, num_actions=3, target_update_frequency=10):
        self.policy_net = DQN(input_channels, num_actions).to(device)
        self.target_net = DQN(input_channels, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode
        self.target_update_frequency = target_update_frequency
        self.steps = 0
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def get_action(self, state, exploration_strategy, episode):
        with torch.no_grad():
            q_values = self.policy_net(state)
            return exploration_strategy.get_action(q_values[0], episode)
            
    def train_step(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, optimizer, gamma):
        # Compute current Q values
        current_q_values = self.policy_net(batch_states).gather(1, batch_actions.unsqueeze(1))
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(batch_next_states).max(1)[0]
            target_q_values = batch_rewards + (gamma * next_q_values * ~batch_dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.update_target_network()
            
        return loss.item()

def preprocess_state(state):
    # Convert to torch tensor and normalize to [0,1], move to device
    state_tensor = torch.FloatTensor(state).permute(2, 0, 1).to(device) / 255.0
    return state_tensor.unsqueeze(0)  # Add batch dimension

def train_dqn_with_exploration(env, agent, optimizer, exploration_strategy, num_episodes=1000, gamma=0.99, 
                              batch_size=32, buffer_size=10000, visualize=False):
    scores = []
    q_values = []
    training_times = []
    losses = []
    exploration_metrics = []
    
    # Create directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_name = exploration_strategy.__class__.__name__
    results_dir = f"dqn_{strategy_name.lower()}_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    # Initialize target network
    target_model = DQN().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    # Step counter for target network updates
    step_counter = 0
    
    if visualize:
        visualizer = SnakeVisualizer(env.width, env.height)
    
    # Initial Population (Warm Start): Fill buffer with random experiences
    print(f"Warming up replay buffer with {min_buffer_size} experiences...")
    state, _, done, _ = env.reset()
    
    while len(replay_buffer) < min_buffer_size:
        # Use random action for warm start
        action = np.random.randint(-1, 2)
        next_state, reward, done, _ = env.step(action)
        
        # Store experience in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        
        state = next_state if not done else env.reset()[0]
        
        if len(replay_buffer) % 100 == 0:
            print(f"Buffer filled: {len(replay_buffer)}/{min_buffer_size}")
    
    print(f"Warm-up complete! Buffer size: {len(replay_buffer)}")
    
    for episode in range(num_episodes):
        state, _, done, _ = env.reset()
        episode_score = 0
        episode_q_values = []
        episode_losses = []
        start_time = time.time()
        
        while not done:
            if visualize:
                if not visualizer.handle_events():
                    return scores, q_values, training_times, losses, exploration_metrics
                visualizer.draw_board(state)
                time.sleep(0.1)  # Slow down visualization
            
            # Preprocess state
            state_tensor = preprocess_state(state)
            
            # Get Q-values
            current_q_values = agent.policy_net(state_tensor)
            episode_q_values.append(current_q_values.mean().item())
            
            # Choose action using exploration strategy
            action = agent.get_action(state_tensor, exploration_strategy, episode)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            episode_score += reward
            
            # Store experience in replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Increment step counter
            step_counter += 1
            
            # Train on batch if buffer has enough samples
            if replay_buffer.can_sample(batch_size):
                # Sample batch from replay buffer
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
                
                # Convert to tensors and move to device
                batch_states = torch.FloatTensor(batch_states).permute(0, 3, 1, 2).to(device) / 255.0
                batch_actions = torch.LongTensor(batch_actions).to(device) + 1  # Convert to 0, 1, 2
                batch_rewards = torch.FloatTensor(batch_rewards).to(device)
                batch_next_states = torch.FloatTensor(batch_next_states).permute(0, 3, 1, 2).to(device) / 255.0
                batch_dones = torch.BoolTensor(batch_dones).to(device)
                
                # Train step
                loss = agent.train_step(batch_states, batch_actions, batch_rewards, 
                                      batch_next_states, batch_dones, optimizer, gamma)
                episode_losses.append(loss)
            
            state = next_state
        
        training_time = time.time() - start_time
        scores.append(episode_score)
        q_values.append(np.mean(episode_q_values))
        training_times.append(training_time)
        losses.append(np.mean(episode_losses) if episode_losses else 0)
        
        # Record exploration metric
        if isinstance(exploration_strategy, EpsilonGreedyStrategy):
            exploration_metrics.append(exploration_strategy.get_epsilon())
        else:  # BoltzmannStrategy
            exploration_metrics.append(exploration_strategy.get_temperature())
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {episode_score}, "
                  f"Avg Q-value: {np.mean(episode_q_values):.2f}, "
                  f"Exploration: {exploration_metrics[-1]:.4f}, "
                  f"Loss: {np.mean(episode_losses) if episode_losses else 0:.4f}, "
                  f"Time: {training_time:.2f}s")
            
            # Save metrics every 10 episodes
            metrics = {
                'episode': episode,
                'score': episode_score,
                'avg_q_value': float(np.mean(episode_q_values)),
                'exploration': float(exploration_metrics[-1]),
                'avg_loss': float(np.mean(episode_losses) if episode_losses else 0),
                'training_time': float(training_time),
                'buffer_size': len(replay_buffer),
                'step_counter': step_counter
            }
            with open(os.path.join(results_dir, 'metrics.json'), 'a') as f:
                json.dump(metrics, f)
                f.write('\n')
    
    if visualize:
        pygame.quit()
    
    # Save final model
    torch.save(agent.policy_net.state_dict(), os.path.join(results_dir, 'model.pth'))
    
    # Create and save plots
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 5, 1)
    plt.plot(scores)
    plt.title(f'Training Scores ({strategy_name})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(os.path.join(results_dir, 'scores.png'))
    
    plt.subplot(1, 5, 2)
    plt.plot(q_values)
    plt.title(f'Average Q-values ({strategy_name})')
    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.savefig(os.path.join(results_dir, 'q_values.png'))
    
    plt.subplot(1, 5, 3)
    plt.plot(losses)
    plt.title(f'Training Loss ({strategy_name})')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(results_dir, 'losses.png'))
    
    plt.subplot(1, 5, 4)
    plt.plot(training_times)
    plt.title(f'Training Times ({strategy_name})')
    plt.xlabel('Episode')
    plt.ylabel('Time (s)')
    plt.savefig(os.path.join(results_dir, 'training_times.png'))
    
    plt.subplot(1, 5, 5)
    plt.plot(exploration_metrics)
    plt.title(f'Exploration Schedule ({strategy_name})')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon/Temperature')
    plt.savefig(os.path.join(results_dir, 'exploration.png'))
    
    plt.close()
    
    return scores, q_values, training_times, losses, exploration_metrics

def compare_target_network():
    # Initialize environment
    env = SnakeGame(32, 32)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"target_network_comparison_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Train with target network
    print("\nTraining with target network...")
    target_agent = DQNAgent(target_update_frequency=10)
    target_optimizer = optim.Adam(target_agent.policy_net.parameters(), lr=0.001)
    target_strategy = EpsilonGreedyStrategy(initial_epsilon=1.0, final_epsilon=0.01)
    
    target_results = train_dqn_with_exploration(
        env, target_agent, target_optimizer, 
        target_strategy, num_episodes=1000, visualize=False
    )
    
    # Train without target network (using policy network for both current and next Q-values)
    print("\nTraining without target network...")
    no_target_agent = DQNAgent(target_update_frequency=float('inf'))  # Never update target network
    no_target_optimizer = optim.Adam(no_target_agent.policy_net.parameters(), lr=0.001)
    no_target_strategy = EpsilonGreedyStrategy(initial_epsilon=1.0, final_epsilon=0.01)
    
    no_target_results = train_dqn_with_exploration(
        env, no_target_agent, no_target_optimizer,
        no_target_strategy, num_episodes=1000, visualize=False
    )
    
    # Compare results
    plt.figure(figsize=(15, 10))
    
    # Plot scores
    plt.subplot(2, 2, 1)
    plt.plot(target_results[0], label='With Target Network')
    plt.plot(no_target_results[0], label='Without Target Network')
    plt.title('Training Scores Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot Q-values
    plt.subplot(2, 2, 2)
    plt.plot(target_results[1], label='With Target Network')
    plt.plot(no_target_results[1], label='Without Target Network')
    plt.title('Average Q-values Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.legend()
    
    # Plot losses
    plt.subplot(2, 2, 3)
    plt.plot(target_results[3], label='With Target Network')
    plt.plot(no_target_results[3], label='Without Target Network')
    plt.title('Training Loss Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training times
    plt.subplot(2, 2, 4)
    plt.plot(target_results[2], label='With Target Network')
    plt.plot(no_target_results[2], label='Without Target Network')
    plt.title('Training Time Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Time (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'target_network_comparison.png'))
    plt.close()
    
    # Save comparison metrics
    comparison_metrics = {
        'with_target': {
            'final_score': float(np.mean(target_results[0][-100:])),
            'final_q_value': float(np.mean(target_results[1][-100:])),
            'final_loss': float(np.mean(target_results[3][-100:])),
            'total_training_time': float(np.sum(target_results[2]))
        },
        'without_target': {
            'final_score': float(np.mean(no_target_results[0][-100:])),
            'final_q_value': float(np.mean(no_target_results[1][-100:])),
            'final_loss': float(np.mean(no_target_results[3][-100:])),
            'total_training_time': float(np.sum(no_target_results[2]))
        }
    }
    
    with open(os.path.join(results_dir, 'target_network_comparison_metrics.json'), 'w') as f:
        json.dump(comparison_metrics, f, indent=4)
    
    # Print comparison results
    print("\nComparison Results:")
    print("\nWith Target Network:")
    print(f"Final Score: {comparison_metrics['with_target']['final_score']:.2f}")
    print(f"Final Q-value: {comparison_metrics['with_target']['final_q_value']:.2f}")
    print(f"Final Loss: {comparison_metrics['with_target']['final_loss']:.4f}")
    print(f"Total Training Time: {comparison_metrics['with_target']['total_training_time']:.2f}s")
    
    print("\nWithout Target Network:")
    print(f"Final Score: {comparison_metrics['without_target']['final_score']:.2f}")
    print(f"Final Q-value: {comparison_metrics['without_target']['final_q_value']:.2f}")
    print(f"Final Loss: {comparison_metrics['without_target']['final_loss']:.4f}")
    print(f"Total Training Time: {comparison_metrics['without_target']['total_training_time']:.2f}s")

def main():
    compare_target_network()

if __name__ == "__main__":
    main() 