# Snake Game DQN: Baseline Implementation

This repository contains a Deep Q-Network (DQN) implementation for the classic Snake game, serving as a baseline for further enhancements such as Experience Replay and Target Networks.

## Deliverables

### 1. Code Implementation

- **Snake Game Environment:**  
  A custom implementation of the Snake game in `snake_game.py`.
- **DQN Agent:**  
  A convolutional neural network agent for learning to play Snake, implemented in `snake_dqn.py`.
- **Visualization:**  
  Real-time game visualization using Pygame (`snake_visualizer.py`) and experiment plotting with Matplotlib.
- **Heuristic Policy:**  
  A simple, rule-based policy for comparison with the DQN agent.

### 2. Brief Presentation

The presentation (to be provided separately) will cover:

- **Neural Network Architecture & Effectiveness:**  
  - Description of the convolutional neural network used as the Q-network.
  - Discussion of why this architecture is suitable for the Snake game and its observed effectiveness.

- **Q-Learning Loss Function Implementation:**  
  - Details on the Q-learning update rule and loss calculation.
  - Explanation of how the loss is computed and used to update the network.

- **Training Experiments & Baseline Performance Metrics:**  
  - Results from training the DQN agent, including:
    - Learning curves (scores, Q-values, training times).
    - Baseline performance metrics (average scores, convergence speed).
  - All metrics and plots are saved automatically during training for reproducibility.

- **Heuristic Policy Algorithm & Performance vs. Random Play:**  
  - Description of the heuristic policy logic.
  - Quantitative comparison of heuristic, random, and DQN agent performance.

---

## Getting Started

### Requirements

- Python 3.7+
- `numpy`
- `torch`
- `pygame`
- `matplotlib`

Install dependencies with:
```bash
pip install -r requirements.txt
```

### Running the Code

Train and evaluate the DQN agent:
```bash
python snake_dqn.py
```
- Training metrics, plots, and the trained model will be saved in a timestamped results directory.

### Files

- `snake_game.py` — Snake game environment.
- `snake_dqn.py` — DQN agent, training loop, evaluation, and metrics saving.
- `snake_visualizer.py` — Pygame-based visualization for the Snake game.
- `game_demo.py` — Example of running the game and plotting with Matplotlib.

---

## Results

- Training and evaluation results are saved as plots and JSON files in the results directory.
- Use these for comparison with enhanced DQN variants (e.g., with Experience Replay and Target Networks).

---

## License

MIT License 