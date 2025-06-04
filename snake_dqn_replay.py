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

def preprocess_state(state):
    # Convert to torch tensor and normalize to [0,1], move to device
    state_tensor = torch.FloatTensor(state).permute(2, 0, 1).to(device) / 255.0
    return state_tensor.unsqueeze(0)  # Add batch dimension

def train_dqn_with_replay(env, model, optimizer, num_episodes=1000, gamma=0.99, 
                         batch_size=32, buffer_size=10000, target_update_freq=100, 
                         min_buffer_size=1000, visualize=False):
    scores = []
    q_values = []
    training_times = []
    losses = []
    
    # Create directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"dqn_replay_results_{timestamp}"
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
                    return scores, q_values, training_times, losses
                visualizer.draw_board(state)
                time.sleep(0.1)  # Slow down visualization
            
            # Preprocess state
            state_tensor = preprocess_state(state)
            
            # Get Q-values
            current_q_values = model(state_tensor)
            episode_q_values.append(current_q_values.mean().item())
            
            # Choose action using epsilon-greedy policy (decaying epsilon)
            epsilon = max(0.01, 0.1 - episode * 0.00009)  # Decay from 0.1 to 0.01
            if np.random.random() < epsilon:
                action = np.random.randint(-1, 2)
            else:
                action = torch.argmax(current_q_values).item() - 1
            
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
                
                # Compute current Q values
                current_q_values = model(batch_states).gather(1, batch_actions.unsqueeze(1))
                
                # Compute next Q values using TARGET NETWORK
                with torch.no_grad():
                    next_q_values = target_model(batch_next_states).max(1)[0]
                    target_q_values = batch_rewards + (gamma * next_q_values * ~batch_dones)
                
                # Compute loss
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                episode_losses.append(loss.item())
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update target network periodically
                if step_counter % target_update_freq == 0:
                    target_model.load_state_dict(model.state_dict())
                    print(f"Target network updated at step {step_counter}")
            
            state = next_state
        
        training_time = time.time() - start_time
        scores.append(episode_score)
        q_values.append(np.mean(episode_q_values))
        training_times.append(training_time)
        losses.append(np.mean(episode_losses) if episode_losses else 0)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {episode_score}, "
                  f"Avg Q-value: {np.mean(episode_q_values):.2f}, "
                  f"Epsilon: {epsilon:.4f}, Loss: {np.mean(episode_losses) if episode_losses else 0:.4f}, "
                  f"Time: {training_time:.2f}s, Buffer: {len(replay_buffer)}")
            
            # Save metrics every 10 episodes
            metrics = {
                'episode': episode,
                'score': episode_score,
                'avg_q_value': float(np.mean(episode_q_values)),
                'epsilon': float(epsilon),
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
    
    # Save final model and target model
    torch.save(model.state_dict(), os.path.join(results_dir, 'model.pth'))
    torch.save(target_model.state_dict(), os.path.join(results_dir, 'target_model.pth'))
    
    # Create and save plots
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(scores)
    plt.title('Training Scores (with Experience Replay)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(os.path.join(results_dir, 'scores.png'))
    
    plt.subplot(1, 4, 2)
    plt.plot(q_values)
    plt.title('Average Q-values (with Experience Replay)')
    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.savefig(os.path.join(results_dir, 'q_values.png'))
    
    plt.subplot(1, 4, 3)
    plt.plot(losses)
    plt.title('Training Loss (with Experience Replay)')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(results_dir, 'losses.png'))
    
    plt.subplot(1, 4, 4)
    plt.plot(training_times)
    plt.title('Training Times (with Experience Replay)')
    plt.xlabel('Episode')
    plt.ylabel('Time (s)')
    plt.savefig(os.path.join(results_dir, 'training_times.png'))
    
    plt.close()
    
    # Save final metrics
    final_metrics = {
        'final_score': float(np.mean(scores[-100:])),  # Last 100 episodes
        'final_q_value': float(np.mean(q_values[-100:])),
        'final_loss': float(np.mean(losses[-100:])),
        'total_training_time': float(np.sum(training_times)),
        'num_episodes': num_episodes,
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'target_update_freq': target_update_freq,
        'min_buffer_size': min_buffer_size,
        'total_steps': step_counter
    }
    with open(os.path.join(results_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    return scores, q_values, training_times, losses

def evaluate_model(model, env, num_episodes=100, visualize=False):
    """Evaluate the trained model over multiple episodes"""
    scores = []
    visualizer = SnakeVisualizer(env.width, env.height) if visualize else None
    
    for episode in range(num_episodes):
        state, _, done, _ = env.reset()
        episode_score = 0
        
        while not done:
            if visualize:
                if not visualizer.handle_events():
                    break
                visualizer.draw_board(state)
                time.sleep(0.1)
            
            # Get action from model
            state_tensor = preprocess_state(state)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values).item() - 1
            
            # Take action
            state, reward, done, _ = env.step(action)
            episode_score += reward
        
        scores.append(episode_score)
        print(f"Evaluation Episode {episode}, Score: {episode_score}")
    
    if visualize:
        pygame.quit()
    
    return scores

def main():
    # Initialize environment and model
    env = SnakeGame(32, 32)
    model = DQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train DQN with Experience Replay
    print("Training DQN with Experience Replay and Target Network...")
    scores, q_values, training_times, losses = train_dqn_with_replay(
        env, model, optimizer, 
        num_episodes=1000, 
        batch_size=32,
        buffer_size=10000,
        target_update_freq=100,  # Update target network every 100 steps
        min_buffer_size=1000,    # Warm start with 1000 experiences
        visualize=False
    )
    
    # Evaluate trained model
    print("\nEvaluating trained model...")
    eval_scores = evaluate_model(model, env, num_episodes=100, visualize=False)
    
    # Print results
    print("\nResults:")
    print(f"DQN with Replay Average Score: {np.mean(scores):.2f}")
    print(f"DQN with Replay Average Q-value: {np.mean(q_values):.2f}")
    print(f"DQN with Replay Average Loss: {np.mean(losses):.4f}")
    print(f"DQN with Replay Average Training Time: {np.mean(training_times):.2f}s")
    print(f"Evaluation Average Score: {np.mean(eval_scores):.2f}")

if __name__ == "__main__":
    main() 