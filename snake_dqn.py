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
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def preprocess_state(state):
    # Convert to torch tensor and normalize to [0,1], move to device
    state_tensor = torch.FloatTensor(state).permute(2, 0, 1).to(device) / 255.0
    return state_tensor.unsqueeze(0)  # Add batch dimension

def heuristic_policy(game):
    """Simple heuristic policy that moves towards the apple while avoiding walls and self"""
    score, apples, head, tail, direction = game.get_state()
    
    if not apples:  # No apples on the board
        return 0  # Continue straight
    
    # Get the closest apple
    apple = apples[0]
    head_y, head_x = head
    
    # Calculate desired direction to apple
    desired_direction = None
    if apple[0] < head_y:  # Apple is above
        desired_direction = 0  # Up
    elif apple[0] > head_y:  # Apple is below
        desired_direction = 2  # Down
    elif apple[1] < head_x:  # Apple is left
        desired_direction = 3  # Left
    elif apple[1] > head_x:  # Apple is right
        desired_direction = 1  # Right
    
    # Check if turning would cause collision
    next_pos = None
    if desired_direction == 0:  # Up
        next_pos = (head_y - 1, head_x)
    elif desired_direction == 1:  # Right
        next_pos = (head_y, head_x + 1)
    elif desired_direction == 2:  # Down
        next_pos = (head_y + 1, head_x)
    elif desired_direction == 3:  # Left
        next_pos = (head_y, head_x - 1)
    
    # Check for collisions
    if (next_pos[0] < 0 or next_pos[0] >= game.height or
        next_pos[1] < 0 or next_pos[1] >= game.width or
        next_pos in tail):
        return 0  # Continue straight if turning would cause collision
    
    # Determine action based on current direction and desired direction
    if direction == desired_direction:
        return 0  # Continue straight
    elif (direction + 1) % 4 == desired_direction:
        return 1  # Turn right
    else:
        return -1  # Turn left

def train_dqn(env, model, optimizer, num_episodes=1000, gamma=0.99, visualize=False):
    scores = []
    q_values = []
    training_times = []
    
    # Create directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"dqn_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    if visualize:
        visualizer = SnakeVisualizer(env.width, env.height)
    
    for episode in range(num_episodes):
        state, _, done, _ = env.reset()
        episode_score = 0
        episode_q_values = []
        start_time = time.time()
        
        while not done:
            if visualize:
                if not visualizer.handle_events():
                    return scores, q_values, training_times
                visualizer.draw_board(state)
                time.sleep(0.1)  # Slow down visualization
            
            # Preprocess state
            state_tensor = preprocess_state(state)
            
            # Get Q-values
            current_q_values = model(state_tensor)
            episode_q_values.append(current_q_values.mean().item())
            
            # Choose action using epsilon-greedy policy
            if np.random.random() < 0.1:  # 10% random actions
                action = np.random.randint(-1, 2)
            else:
                action = torch.argmax(current_q_values).item() - 1
            
            # Take action
            next_state, reward, done, info = env.step(action)
            episode_score += reward
            
            # Update model
            next_state_tensor = preprocess_state(next_state)
            with torch.no_grad():  # Only use no_grad for next state computation
                next_q_values = model(next_state_tensor)
            target_q = reward + gamma * torch.max(next_q_values) * (not done)
            
            current_q = current_q_values[0][action + 1]
            loss = nn.MSELoss()(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
        
        training_time = time.time() - start_time
        scores.append(episode_score)
        q_values.append(np.mean(episode_q_values))
        training_times.append(training_time)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {episode_score}, "
                  f"Avg Q-value: {np.mean(episode_q_values):.2f}, "
                  f"Time: {training_time:.2f}s")
            
            # Save metrics every 10 episodes
            metrics = {
                'episode': episode,
                'score': episode_score,
                'avg_q_value': float(np.mean(episode_q_values)),
                'training_time': float(training_time)
            }
            with open(os.path.join(results_dir, 'metrics.json'), 'a') as f:
                json.dump(metrics, f)
                f.write('\n')
    
    if visualize:
        pygame.quit()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(results_dir, 'model.pth'))
    
    # Create and save plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(os.path.join(results_dir, 'scores.png'))
    
    plt.subplot(1, 3, 2)
    plt.plot(q_values)
    plt.title('Average Q-values')
    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.savefig(os.path.join(results_dir, 'q_values.png'))
    
    plt.subplot(1, 3, 3)
    plt.plot(training_times)
    plt.title('Training Times')
    plt.xlabel('Episode')
    plt.ylabel('Time (s)')
    plt.savefig(os.path.join(results_dir, 'training_times.png'))
    
    plt.close()
    
    # Save final metrics
    final_metrics = {
        'final_score': float(np.mean(scores[-100:])),  # Last 100 episodes
        'final_q_value': float(np.mean(q_values[-100:])),
        'total_training_time': float(np.sum(training_times)),
        'num_episodes': num_episodes
    }
    with open(os.path.join(results_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    return scores, q_values, training_times

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
    
    # Train DQN with visualization
    print("Training DQN...")
    scores, q_values, training_times = train_dqn(env, model, optimizer, visualize=False)
    
    # Evaluate trained model
    print("\nEvaluating trained model...")
    eval_scores = evaluate_model(model, env, num_episodes=100, visualize=False)
    
    # Print results
    print("\nResults:")
    print(f"DQN Average Score: {np.mean(scores):.2f}")
    print(f"DQN Average Q-value: {np.mean(q_values):.2f}")
    print(f"DQN Average Training Time: {np.mean(training_times):.2f}s")
    print(f"Evaluation Average Score: {np.mean(eval_scores):.2f}")

if __name__ == "__main__":
    main() 