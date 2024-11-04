# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
# Replace 'wine_dataset.csv' with the path to your local dataset
df = pd.read_csv("wine-clustering.csv")

# Normalize the dataset
def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

normalized_data = normalize_data(df.values)

# Define RL parameters
num_actions = 3  # Example actions: 0, 1, 2 (for different types of recommendations)
num_states = 5   # Discretize the state space into 5 states
num_episodes = 1000
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

# Initialize Q-tables for SARSA and Q-learning
q_table_sarsa = np.zeros((num_states, num_actions))
q_table_qlearning = np.zeros((num_states, num_actions))

# Function to discretize the state
def discretize_state(state):
    # Ensure state values stay within the valid range for indexing
    state_idx = np.clip(np.digitize(state, bins=np.linspace(0, 1, num_states + 1)) - 1, 0, num_states - 1)
    return int(np.mean(state_idx))  # Return an average index if multiple dimensions

# Choose an action based on epsilon-greedy strategy
def choose_action(state, q_table):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# Train the SARSA agent
def train_sarsa(normalized_data):
    for episode in range(num_episodes):
        # Start from a random state
        state = normalized_data[np.random.randint(len(normalized_data))]
        state_idx = discretize_state(state)
        action = choose_action(state_idx, q_table_sarsa)

        while True:
            # Simulate environment feedback (reward)
            reward = np.random.rand()  # Example reward function

            # Select next state and action
            next_state = normalized_data[np.random.randint(len(normalized_data))]
            next_state_idx = discretize_state(next_state)
            next_action = choose_action(next_state_idx, q_table_sarsa)

            # Update Q-value using SARSA update rule
            current_q = q_table_sarsa[state_idx, action]
            next_q = q_table_sarsa[next_state_idx, next_action]
            q_table_sarsa[state_idx, action] += learning_rate * (reward + discount_factor * next_q - current_q)

            state_idx = next_state_idx
            action = next_action

            # Break if a certain condition is met
            if np.random.rand() < 0.1:  # Random condition to end the loop
                break

# Train the Q-learning agent
def train_qlearning(normalized_data):
    for episode in range(num_episodes):
        # Start from a random state
        state = normalized_data[np.random.randint(len(normalized_data))]
        state_idx = discretize_state(state)

        while True:
            # Choose action
            action = choose_action(state_idx, q_table_qlearning)

            # Simulate environment feedback (reward)
            reward = np.random.rand()  # Example reward function

            # Select next state
            next_state = normalized_data[np.random.randint(len(normalized_data))]
            next_state_idx = discretize_state(next_state)

            # Update Q-value using Q-learning update rule
            current_q = q_table_qlearning[state_idx, action]
            max_next_q = np.max(q_table_qlearning[next_state_idx])
            q_table_qlearning[state_idx, action] += learning_rate * (reward + discount_factor * max_next_q - current_q)

            state_idx = next_state_idx

            # Break if a certain condition is met
            if np.random.rand() < 0.1:  # Random condition to end the loop
                break

# Execute training for SARSA and Q-learning
train_sarsa(normalized_data)
train_qlearning(normalized_data)

# Display the Q-tables
plt.figure(figsize=(12, 5))

# Q-table for SARSA
plt.subplot(1, 2, 1)
plt.imshow(q_table_sarsa, cmap='hot', interpolation='nearest')
plt.title('Q-Table Heatmap (SARSA)')
plt.xlabel('Action')
plt.ylabel('State')
plt.colorbar()

# Q-table for Q-learning
plt.subplot(1, 2, 2)
plt.imshow(q_table_qlearning, cmap='hot', interpolation='nearest')
plt.title('Q-Table Heatmap (Q-learning)')
plt.xlabel('Action')
plt.ylabel('State')
plt.colorbar()

plt.show()
