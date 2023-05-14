import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr, self.size = 0, 0

        self.state_buf = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_state_buf[self.ptr] = next_state
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        states = self.state_buf[idxs]
        actions = self.action_buf[idxs]
        rewards = self.reward_buf[idxs]
        next_states = self.next_state_buf[idxs]
        dones = self.done_buf[idxs]
        return (
            torch.as_tensor(states),
            torch.as_tensor(actions),
            torch.as_tensor(rewards),
            torch.as_tensor(next_states),
            torch.as_tensor(dones),
        )

    def __len__(self):
        return self.size

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        return self.max_action * torch.tanh(self.fc3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = F.relu(self.fc1(torch.cat([state, action], 1)))
        q = F.relu(self.fc2(q))
        return self.fc3(q)

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)

        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.001)

        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.001)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

        self.memory = ReplayBuffer(state_dim, action_dim)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size):
        self.total_it += 1

        # Sample a batch of transitions from memory
        state, action, reward, next_state, not_done = self.memory.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(1 - not_done).to(self.device)

        # Select next action according to target policy
        next_action = self.actor_target(next_state)

        # Add Gaussian noise to target action (for exploration)
        noise = torch.FloatTensor(next_action.shape).data.normal_(0, self.policy_noise).to(self.device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q-values
        target_Q1 = self.critic1_target(next_state, next_action)
        target_Q2 = self.critic2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Compute current Q-values
        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)

        # Compute critic loss
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic2.state_dict(), filename + "_critic2")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_target.load_state_dict(torch.load(filename + "_actor"))
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic1_target.load_state_dict(torch.load(filename + "_critic1"))
        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        self.critic2_target.load_state_dict(torch.load(filename + "_critic2"))

env = gym.make('LunarLanderContinuous-v2')
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
