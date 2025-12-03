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
from datetime import datetime
import os
from poker_agents import *

def run_tournament(env, agents, evaluate=False):

    obs, _ = env.reset(options={"reset_button": True, "reset_stacks": True})
    prev_stacks = obs['stacks'].copy()
    done = False

    while not done:

        player_idx = env.table.dealer.action
        curr_agent = agents[player_idx]

        if isinstance(curr_agent, (QAgent, DQNAgent)):
            state = curr_agent._preprocess_state(obs)
            action = curr_agent.act(state)
        else:
            action = curr_agent.act(obs)

        next_obs, reward, done, _, _ = env.step(action)

        # Only compute next state for SAME PLAYER
        next_state = None
        if next_obs is not None and isinstance(curr_agent, (QAgent, DQNAgent)):
            next_state = curr_agent._preprocess_state(next_obs)

        # reward shaping
        new_stacks = next_obs['stacks'].copy()
        stack_diff = [new_stacks[i] - prev_stacks[i] for i in range(env.num_players)]

        if not done:
            reward = stack_diff
            reward = [min(r // env.table.dealer.blinds[1], env.max_stack_bb - 1) for r in reward]
            reward = [np.tanh(r / 5.0) for r in reward]

        # Training step
        if not evaluate:
            for i, agent in enumerate(agents):
                if isinstance(agent, (QAgent, DQNAgent)):
                    agent.update_parameters(
                        state if i == player_idx else None,
                        action,
                        reward[i],
                        next_state if i == player_idx else None,
                        done
                    )

        prev_stacks = new_stacks
        obs = next_obs

    return reward

""" must provide a fixed lineup """
def run_n_tournaments(env, n_tournaments, evaluate=False, fixed_lineup=None):

    rewards_per_tournament = []

    if evaluate:
        for agent in fixed_lineup:
            if hasattr(agent, "epsilon"):
                agent.epsilon = 0.0

    for tournament_idx in tqdm(range(n_tournaments), desc="Running Tournament", ascii=True, ncols=80):

        reward = run_tournament(env, fixed_lineup, evaluate)

        if evaluate:
            rewards_per_tournament.append(reward)

    return rewards_per_tournament


""" must provide a list of agents forming the base of the training pool """
def training_table(env, agents, pool):
    for _ in range(env.num_players - len(agents)) :
        Opponent = random.choice(pool)
        agents.append(Opponent(env))
    return agents


def train(env, n_tournaments, lineup):
    run_n_tournaments(env, n_tournaments, evaluate=False, fixed_lineup=lineup)



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
    
def save_fig(fig, name=None, directory="plots"):
    """Save a matplotlib figure into a directory and close it."""
    os.makedirs(directory, exist_ok=True)

    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"fig_{timestamp}.png"

    path = os.path.join(directory, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

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


def train_and_evaluate(env, n_tournaments_learn, freq_eval, training_lineup, evaluation_lineups):
    n_evaluations = n_tournaments_learn // freq_eval
    window_size = max(50, freq_eval // 20)
    fig, axes = plt.subplots(
        n_evaluations, 
        len(evaluation_lineups), 
        figsize=(50, 12 * n_evaluations), 
        sharex='col')    
    if n_evaluations == 1:
        axes = np.array([axes])
    if len(evaluation_lineups) == 1:
        axes = np.expand_dims(axes, axis=1)
    for eval_idx in range(n_evaluations):
        train(env, freq_eval, training_lineup)
        for ax, lineup in zip(axes[eval_idx], evaluation_lineups):
            rewards_per_tournament = run_n_tournaments(
                env, freq_eval, evaluate=True, fixed_lineup=lineup)
            plot_results(rewards_per_tournament, lineup,
                         freq_eval, window_size, ax)
    plt.tight_layout()
    save_fig(fig, directory="eval_logs")


def main():

    env = simulation.PokerTournament()

    dqn1 = DQNAgent(env, "dqn1")

    dqn2 = DQNAgent(env, "dqn2")

    n_tournaments_learn = 1000
    RANDOM_LINEUP = [dqn1,RandomAllInFoldAgent(env), RandomAllInFoldAgent(env), RandomAllInFoldAgent(env)]
    ALL_IN_PAIR_LINEUP = [dqn1, AllInPairAgent(
        env), AllInPairAgent(env), AllInPairAgent(env)]

    DQN_LINEUP = [dqn1, dqn2, RandomAllInFoldAgent(
        env), RandomAllInFoldAgent(env)]

    train_and_evaluate(env, n_tournaments_learn, freq_eval=500,
                       training_lineup=DQN_LINEUP,
                       evaluation_lineups=[DQN_LINEUP])
   
    env.close()
    
if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #s = io.StringIO()
    #pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(40)
    #print(s.getvalue())
