import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import math
import matplotlib.pyplot as plt
import os


class QLearningAgent:
    def __init__(self, action_size, state_sizes, bins_per_dimension):
        self.state_sizes = state_sizes
        self.bins_per_dimension = bins_per_dimension
        total_states = math.prod(bins_per_dimension.values())
        print(total_states)
        self.q_table = np.zeros((total_states, action_size))
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.max_exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.01

    def state_to_index(self, state):
        index = 0
        multiplier = 1
        for state_var, bins in reversed(list(self.bins_per_dimension.items())):
            index += state[state_var] * multiplier
            multiplier *= self.bins_per_dimension[state_var]
        return index

    def choose_action(self, state):
        state_index = self.state_to_index(state)
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(0, len(self.q_table[state_index]))
        else:
            action = np.argmax(self.q_table[state_index])
        return action

    def update_q_table(self, state, action, reward, new_state):
        state_index = self.state_to_index(state)
        new_state_index = self.state_to_index(new_state)
        best_next_action = np.max(self.q_table[new_state_index])
        self.q_table[state_index, action] = self.q_table[state_index, action] + \
            self.learning_rate * (reward + self.discount_factor * best_next_action - self.q_table[state_index, action])

def discretize_state(state, env, bins_per_dimension):
    # Assuming state is a 1D numpy array of continuous values
    discrete_state = {}
    for i, (key, bins) in enumerate(bins_per_dimension.items()):
        range_min = env.observation_space.low[i]
        range_max = env.observation_space.high[i]
        discrete_state[key] = np.digitize(state[i], np.linspace(range_min, range_max, bins)) - 1
    return discrete_state


def train_agent(env, episodes, bins_per_dimension):
    action_size = env.action_space.n  # Assuming discrete actions
    print("Action space size:", action_size)
    print("state size", env.observation_space.shape[0])
    agent = QLearningAgent(action_size=action_size, state_sizes=env.observation_space.shape[0],
                           bins_per_dimension=bins_per_dimension)
    train_rewards = []  # This will now store total rewards for each episode
    for episode in range(episodes):
        print(f"Train episode {episode + 1}")
        raw_state = env.reset()
        state = discretize_state(raw_state, env, bins_per_dimension)  # Discretize initial state
        done = False
        episode_rewards = 0  # Total rewards accumulated during the episode
        step = 0
        while not done:
            print(step)
            step = step + 1
            action = agent.choose_action(state)  # Agent chooses action
            new_raw_state, reward, done, _ = env.step(action)  # Environment processes action
            new_state = discretize_state(new_raw_state, env, bins_per_dimension)  # Discretize new state

            agent.update_q_table(state, action, reward, new_state)  # Update Q-table
            state = new_state  # Move to new state
            episode_rewards += reward  # Accumulate rewards

        train_rewards.append(episode_rewards)  # Store total rewards for the episode

        # Exploration rate decay
        agent.exploration_rate = max(agent.min_exploration_rate,
                                     agent.exploration_rate * np.exp(-agent.exploration_decay_rate * episode))

    plt.plot(train_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

    # Saving training results
    output_dir = r'C:\Users\tessel.kaal\Documents\GitHub\Thesis\Output'
    train_results_path = os.path.join(output_dir, 'train_rewards.csv')
    with open(train_results_path, 'w') as file:
        file.write('\n'.join(map(str, train_rewards)))

    return agent
def evaluate_agent(env, agent, bins_per_dimension, episodes):
    episode_rewards_list = []  # List to store total rewards of each episode
    total_rewards = 0

    for episode in range(episodes):
        print(f"Evaluation episode {episode + 1}")
        raw_state = env.reset()
        state = discretize_state(raw_state, env, bins_per_dimension)
        done = False
        episode_rewards = 0

        while not done:
            state_index = agent.state_to_index(state)
            action = np.argmax(agent.q_table[state_index])
            new_raw_state, reward, done, _ = env.step(action)
            new_state = discretize_state(new_raw_state, env, bins_per_dimension)
            state = new_state
            episode_rewards += reward

        episode_rewards_list.append(episode_rewards)  # Append total reward of the episode to the list
        total_rewards += episode_rewards  # Add the episode's total rewards to the cumulative total

    average_reward = total_rewards / episodes  # Calculate average reward
    print(f"Total Rewards: {total_rewards}, Average Reward: {average_reward}")

    # Plot the total rewards
    plt.plot(episode_rewards_list)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

    # Saving evaluation results
    output_dir = r'C:\Users\tessel.kaal\Documents\GitHub\Thesis\Output'
    eval_results_path = os.path.join(output_dir, 'evaluation_rewards.csv')
    with open(eval_results_path, 'w') as file:
        file.write('\n'.join(map(str, episode_rewards_list)))

    return average_reward, episode_rewards_list