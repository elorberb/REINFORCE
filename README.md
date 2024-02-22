# Project REINFORCE

## Overview
This project focuses on implementing and comparing various reinforcement learning algorithms within the 'CartPole-v1' environment from OpenAI's gym toolkit. The series includes REINFORCE, REINFORCE with baseline, and REINFORCE with actor-critic methods.

## Algorithms
- **REINFORCE**: A basic policy gradient method used for episodic tasks.
- **REINFORCE with Baseline**: Enhances REINFORCE by introducing a baseline for variance reduction.
- **REINFORCE with Actor-Critic**: Combines policy gradients with a value function, acting as the critic for improved performance.

## Environment
The algorithms are tested in the following OpenAI gym environment:
```python
env = gym.make('CartPole-v1')
```
A classic test that requires an agent to balance a pole on a moving cart.
## Comparison Metrics
Performance of the algorithms is compared using:
- Duration of episodes
- Accumulated rewards per episode
- Number of episodes to solve the task
- Learning stability and consistency
## Usage
To run the experiments:
1. Install the `requirements.txt`
2. Run the game runners in `experiments.ipynb`
3. Analyze the results of the experiments in the `results_analysis.ipynb`