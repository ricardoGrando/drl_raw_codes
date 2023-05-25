import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import gym_hydrone
from replay_buffer import ReplayBuffer
from networks import Actor, Critic
from learner import TD3

env = gym.make('hydrone_Circuit_Simple-v0')
env.seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
filename = "whatever"

td3 = TD3(state_dim, action_dim, max_action)

total_timesteps = 10000
timesteps_since_eval = 0
episode_num = 0
episode_reward = 0
eval_freq = 100
num_eval_episodes = 3
batch_size = 128

eval = False

if (not eval):
    for t in range(total_timesteps):

        # Evaluate the policy periodically
        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            avg_reward = 0.0
            for i in range(num_eval_episodes):
                state, done = env.reset(), False
                while not done:
                    action = td3.select_action(state)
                    state, reward, done, _ = env.step(action)
                    avg_reward += reward
            avg_reward /= num_eval_episodes
            print("---------------------------------------")
            print(f"Evaluation Episode: {episode_num}")
            print(f"Avg. Reward: {avg_reward:.3f}")
            print("---------------------------------------")

            td3.save(filename)

        # Collect experience and update the agent
        state, done = env.reset(), False
        while not done:
            action = td3.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # print(done)
            td3.memory.add(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            td3.train(batch_size)

        # Print status
        if done:
            episode_num += 1
            timesteps_since_eval += 1
            print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {len(td3.memory)} Reward: {episode_reward:.3f}")
            episode_reward = 0

if (eval):
    td3.load(filename)

    # Evaluate the policy
    num_episodes = 10
    avg_reward = 0.0

    for i in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Use actor network to select action
            state = torch.FloatTensor(state).to(td3.device)
            # state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = td3.actor(state)
            action = action_tensor.cpu().detach().numpy()
            # print(action)

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Update state for next iteration
            state = next_state

            # Render the environment
            env.render()

        avg_reward += episode_reward

        print(f'Episode {i+1}: reward = {episode_reward:.2f}')

    print(f'Average reward over {num_episodes} episodes: {avg_reward/num_episodes:.2f}')

    env.close()
