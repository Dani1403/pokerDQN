import time
from qagent import QAgent
from dqn_agent import DQNAgent
from simconfig import PRIZE_POOL
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
           #normalize reward to big blind units
           reward = [min(r // env.table.dealer.blinds[1], env.max_stack - 1) for r in reward]
        
        if not evaluate:
            for i, agent in enumerate(agents):
                if isinstance(agent, QAgent):
                    agent.update_parameters(obs, action, reward[i], next_obs, done)

        obs = next_obs
        prev_stacks = new_stacks

    return reward


def training_table(env, q, pool):
    agents = [q]
    for _ in range(env.num_players - 1) :
        Opponent = random.choice(pool)
        agents.append(Opponent(env))

    return agents


""" must provide a fixed lineup OR a training pool """
def run_n_tournaments(env, qagent, n_tournaments, evaluate=False, fixed_lineup=None, training_pool=None):

    rewards_per_tournament = []

    for _ in tqdm(range(n_tournaments), desc="Running Tournament", ascii=True, ncols=80):

        if fixed_lineup:
            agents = fixed_lineup

        else:
            agents = training_table(env, qagent, training_pool)

        if evaluate:
            agents = fixed_lineup
            for agent in agents:
                if hasattr(agent, "epsilon"):
                    agent.epsilon = 0.0

        reward = run_tournament(env, agents, evaluate)

        if evaluate:
            rewards_per_tournament.append(reward)

    return rewards_per_tournament

def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')


def placements(rewards_per_tournament):
    arr = np.array(rewards_per_tournament)
    ranks = np.argsort(np.argsort(-arr, axis=1), axis=1) + 1
    placements = ranks.T
    placement_summary = []
    for i in range(placements.shape[0]):
        unique, counts = np.unique(placements[i], return_counts=True)
        percentages = 100 * counts / placements.shape[1]
        summary = dict(zip(unique, percentages))
        summary_str = " ".join(
            [f"{place}: {percent:.2f}%" for place, percent in summary.items()])
        placement_summary.append(summary_str)

    return placement_summary
    


def plot_results(rewards_per_tournament, agents, n_tournaments, window_size, ax):
    reward_per_agent = np.array(rewards_per_tournament).T
    placement_summary = placements(rewards_per_tournament)

    for i, rewards in enumerate(reward_per_agent):
        smoothed = moving_average(rewards, window_size)
        ax.plot(range(window_size - 1, n_tournaments), smoothed, linewidth=2, label=f"Player {i + 1}")
    ax.set_title("Rewards per Tournament in Poker Simulation")
    ax.set_xlabel("Tournament")
    ax.set_ylabel("Rewards")
    x_ticks_spacing = max(1, n_tournaments // 10)
    ax.set_xticks(ticks = range(0,n_tournaments,x_ticks_spacing))

    avg_rewards = reward_per_agent.mean(axis=1)

    n_agents = min(len(agents), len(avg_rewards), len(placement_summary))   

    ax.legend([f"Player {i + 1} : {agents[i]} \n \
                  Average reward: {avg_rewards[i]:.2f} \n \
                  Placements: {placement_summary[i]}"
               for i in range(n_agents)]),
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.4)


def main():

    env = simulation.PokerTournament()

    q = DQNAgent(env)

    #TRAIN

    n_tournaments_learn = 10000
    RANDOM_LINEUP = [q,RandomAllInFoldAgent(env), RandomAllInFoldAgent(env), RandomAllInFoldAgent(env)]
    run_n_tournaments(env, q, n_tournaments_learn, evaluate=False, training_pool=None, fixed_lineup=RANDOM_LINEUP)


    #EVALUATE

    n_tournaments_evaluate = 500
    window_size = max(50, n_tournaments_evaluate // 20)

    EVALUATION_LINEUPS = [ 
        RANDOM_LINEUP
    ]

    num_lineups = len(EVALUATION_LINEUPS)

    fig, axes = plt.subplots(num_lineups, 1, figsize=(50,12), sharex=True)

    if num_lineups == 1:
        axes = [axes]

    for ax, lineup in zip(axes, EVALUATION_LINEUPS):
        rewards_per_tournament = run_n_tournaments(
                env, q, n_tournaments_evaluate, evaluate=True, fixed_lineup=lineup)

        plot_results(rewards_per_tournament, lineup, n_tournaments_evaluate, window_size, ax)

    plt.tight_layout()
    plt.show()


    env.close()
    
if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #s = io.StringIO()
    #pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(40)
    #print(s.getvalue())
