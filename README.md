# Swarm Reinforcement Learning based on Particle Swarm Optimization 

## Overview

This project implements a collaborative learning framework using multiple Q-learning agents with Particle Swarm Optimization (PSO) equations. The goal is to investigate the behaviour of these agents in a continuous space environment. Unlike the original paper, this implementation focuses on continuous state spaces and utilizes neural networks to replace tabular functions.

### Outline

1. **DQN_PSO_Agent Class:**
   - Manages multiple Q-learning agents using PSO for information exchange.
   - Implements a memory buffer for experience replay.
   - Tracks individual and global best scores and corresponding neural network models.
   - Facilitates the update of Q-values using PSO equations.
2. **DQNAgent Class:**
   - Represents a single Q-learning agent with a neural network model.
   - Implements methods for getting actions, training the model, and updating target models.
4. **CartPole Gym Environment:**
   - Utilizes the CartPole environment from the OpenAI Gym toolkit for experimentation.
   - Agents interact with this environment to learn optimal strategies for pole balancing.

## How to Run the Code

1. **Requirements:**
   - Python (version 3.8.0)
   - Libraries: Gym, NumPy, Pytorch, Matplotlib

2. **Installation:**
   ```bash
   pip install -r requirements.txt
	```

3. **Running the Code:**
   ```bash
   python swarm_dqn.py
	```

   - The script initializes multiple agents and runs episodes in the CartPole environment.
   - Individual and global best models are updated based on agent performance.
   - Q-values are updated using PSO equations after a certain number of episodes.

## Results 

### Experiment 1: Single Agent

Single Agent takes 130 episodes to converge to optimal policy 

![dqn](https://github.com/Srini-Rohan/Swarm-Reinforcement-Learning-Using-PSO/assets/76437900/3cbc4ed6-8c7f-4f40-9d1d-dbca618ad3a9)

### Experiment 2: Multi Agent 

Implemented using 4 agents and one of the agent converges to best optimal policy in 117 episodes
![dqn](https://github.com/Srini-Rohan/Swarm-Reinforcement-Learning-Using-PSO/assets/76437900/9df83590-d221-4efb-a38b-6939c91336ca)
