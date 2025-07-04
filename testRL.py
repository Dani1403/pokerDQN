import gymnasium as gym
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

env = gym.make("Blackjack-v1", natural=True, sab=False)

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, state) -> int:
        return self.env.action_space.sample()

    def update_parameters(self, obs, action, reward, next_obs):
        pass


class StickAt17Agent:
    def __init__(self, env):
        self.env = env

    def get_action(self, state) -> int:
        player_sum, dealer_card, usable_ace = state
        if player_sum >= 17:
            return 0  # Stick
        else:
            return 1  # Hit

    def update_parameters(self, obs, action, reward, next_obs):
        pass



num_possible_sum = 21 - 4 + 1
offset_sum = 4
num_possible_dealer_card = 10 
offset_card = 1
num_options_ace = 2
class QAgent:
    def __init__(self,env, alpha, gamma):
        self.env = env

        self.qtable = np.zeros([num_possible_sum + 10 * offset_sum, num_possible_dealer_card + 2 * offset_card, num_options_ace, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, obs) -> int:
        return np.argmax(self.qtable[ obs[0] , obs[1] , obs[2] ])

    def update_parameters(self, obs, action, reward, next_obs):
            old_value = self.qtable[obs[0] , obs[1] , obs[2], action]
            next_max = np.max(self.qtable[next_obs[0] , next_obs[1] , next_obs[2]])
            new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
            self.qtable[obs[0] , obs[1] , obs[2], action] = new_value

def train(env, agent, n_episodes): 
    env.reset()
    reward_per_episode = []

    epsilon = 1

    for i in tqdm(range(n_episodes), desc="Episodes", unit="episode"):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:

            #exploration vs exploitation
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update_parameters(obs, action, reward, next_obs) 
            done = terminated or truncated
            obs = next_obs
            total_reward += reward

        #epsilon decay 
        epsilon = max(epsilon * 0.999, 0.01)
        if type(agent) == QAgent:
            agent.alpha = max(agent.alpha * 0.999, 0.0001)
        reward_per_episode.append(total_reward)

    return reward_per_episode



# Helper: moving average smoothing
def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

env = gym.make("Blackjack-v1", natural=True, sab=False)
Ragent = RandomAgent(env)
stickagent = StickAt17Agent(env)
Qagent = QAgent(env, 0.01, 1)

n_episodes = 10000

random_rewards = train(Ragent, n_episodes)
sticky_rewards = train(stickagent, n_episodes)
Qrewards = train(Qagent, n_episodes)

# Calculate the moving average
window_size = 1000  
smoothed_random_rewards = moving_average(random_rewards, window_size)
smoothed_sticky_rewards = moving_average(sticky_rewards, window_size)
smoothed_Q_rewards = moving_average(Qrewards, window_size)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(smoothed_random_rewards, label="Random agent")
plt.plot(smoothed_sticky_rewards, label="Stick at 17 agent")
plt.plot(smoothed_Q_rewards, label="Q-learning agent")

plt.axhline(np.mean(random_rewards), color='blue', linestyle='--', label=f"Random Avg: {np.mean(random_rewards):.2f}")
plt.axhline(np.mean(sticky_rewards), color='orange', linestyle='--', label=f"Stick Avg: {np.mean(sticky_rewards):.2f}")
plt.axhline(np.mean(Qrewards), color='green', linestyle='--', label=f"Q-Learning Avg: {np.mean(Qrewards):.2f}")

plt.title(" Blackjack Rewards over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.show()