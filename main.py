import time
from qagent import QAgent
from simconfig import PRIZE_POOL, AGENTS
import simulation                      
import gymnasium as gym      
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cProfile, pstats, io
import sys

from poker_agents import *

def run_tournament(env, agents, evaluate=False):
    obs, _ = env.reset(options={"reset_button": True, "reset_stacks": True})
    prev_stacks = obs['stacks'].copy()
    done = False
    while not done:
        curr_agent = agents[env.table.dealer.action]
        action = curr_agent.act(obs)

        next_obs, reward, done, _, _ = env.step(action)

        new_stacks = next_obs['stacks'].copy()
        stack_diff = [new_stacks[i] - prev_stacks[i]
                      for i in range(env.num_players)]

        if not done:
           reward = stack_diff
        
        if not evaluate:
            for i, agent in enumerate(agents):
                if isinstance(agent, QAgent):
                    agent.update_parameters(obs, action, reward[i], next_obs, done)

        obs = next_obs
        prev_stacks = new_stacks

    return reward

def run_n_tournaments(env, agents, n_tournaments, evaluate=False):
    rewards_per_tournament = []
    for tournament in tqdm(range(n_tournaments), desc="Running Tournament", ascii=True, ncols=80):
        reward = run_tournament(env, agents, evaluate)
        if evaluate:
            rewards_per_tournament.append(reward)
    return rewards_per_tournament

def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

def plot_results(rewards_per_tournament, agents, n_tournaments, window_size):
    reward_per_agent = np.array(rewards_per_tournament).T

    plt.figure(figsize=(50, 12))

    for i, rewards in enumerate(reward_per_agent):
        smoothed = moving_average(rewards, window_size)
        plt.plot(range(window_size - 1, n_tournaments), smoothed, linewidth=2, label=f"Player {i + 1}")
    plt.title("Rewards per Tournament in Poker Simulation")
    plt.xlabel("Tournament")
    plt.ylabel("Rewards")
    x_ticks_spacing = max(1, n_tournaments // 10)
    plt.xticks(ticks = range(0,n_tournaments,x_ticks_spacing), rotation=45)

    avg_rewards = np.mean(rewards_per_tournament, axis=0)
    win_count = [np.count_nonzero(reward_per_agent[i] == PRIZE_POOL[0]) for i in range(len(agents))]
    win_rate = [(win / n_tournaments) * 100 for win in win_count]

    plt.legend([f"Player {i + 1} : {agent} \n \
                  Average reward: {avg_rewards[i]:.2f} \n \
                  Win rate: {win_rate[i]:.2f}%" \
               for i, agent in enumerate(agents)])    
    plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.4)
    plt.show()


def main():
    
    env = simulation.PokerTournament()

    agents = [agent(env) for agent in AGENTS]

    n_tournaments_learn = 100000
    window_size   = 5000

    run_n_tournaments(env, agents, n_tournaments_learn, evaluate=False)

    for a in agents:
        if hasattr(a, "epsilon"):
            a.epsilon = 0.0

    n_tournaments_evaluate = n_tournaments_learn // 5

    rewards_per_tournament = run_n_tournaments(
            env, agents, n_tournaments_evaluate, evaluate=True)

    plot_results(rewards_per_tournament, agents, n_tournaments_evaluate, window_size)

    env.close()
    
if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #s = io.StringIO()
    #pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(40)
    #print(s.getvalue())
