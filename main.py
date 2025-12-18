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
            bb = env.table.dealer.blinds[1]
            cap = env.max_stack_bb - 1
            reward = [int(np.clip(r // bb, -cap, cap)) for r in stack_diff]

        # Training step
        if not evaluate:
            for i, agent in enumerate(agents):
                if i == player_idx and isinstance(agent, (QAgent, DQNAgent)):
                    agent.update_parameters(
                        state,
                        action,
                        reward[i],
                        next_state,
                        done
                    )

        prev_stacks = new_stacks
        obs = next_obs

    return reward

""" must provide a fixed lineup """
def run_n_tournaments(env, n_tournaments, evaluate=False, fixed_lineup=None, desc=None):

    rewards_per_tournament = []
    saved_epsilons = {}

    if evaluate:
        for agent in fixed_lineup:
            if hasattr(agent, "epsilon"):
                saved_epsilons[agent] = agent.epsilon
                agent.epsilon = 0.0

    for tournament_idx in tqdm(range(n_tournaments), ascii=True, ncols=80, desc=desc):

        reward = run_tournament(env, fixed_lineup, evaluate)

        if evaluate:
            rewards_per_tournament.append(reward)

    if evaluate:
        for agent, eps in saved_epsilons.items():
            agent.epsilon = eps

    return rewards_per_tournament


""" must provide a list of agents forming the base of the training pool """
def training_table(env, agents, pool):
    for _ in range(env.num_players - len(agents)) :
        Opponent = random.choice(pool)
        agents.append(Opponent(env))
    return agents


def train(env, n_tournaments, lineup, desc):
    run_n_tournaments(env, n_tournaments, evaluate=False, fixed_lineup=lineup, desc=desc)


def evaluate(env, n_tournaments, lineup, desc):
    rewards_per_tournament = run_n_tournaments(
        env, n_tournaments, evaluate=True, fixed_lineup=lineup, desc=desc)
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
        if len(rewards) < window_size:
            smoothed = rewards
            x_values = range(len(rewards))
        else:
            smoothed = moving_average(rewards, window_size)
            x_values = range(window_size - 1, window_size - 1 + len(smoothed))

        ax.plot(x_values, smoothed, linewidth=2, label=f"Player {i + 1}")

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


def train_and_evaluate(env, N_total, learn_size, eval_size, training_lineup, evaluation_lineups):
    n_evaluations = N_total // learn_size
    window_size = max(50, eval_size // 20)
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
        train(env, learn_size, training_lineup, desc=f"Running training session {eval_idx+1}")
        if eval_idx == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for agent in training_lineup:
            if hasattr(agent, "save"):
                agent.save(f"checkpoints/{agent}/{timestamp}/iter_{eval_idx+1}.pt")
                if eval_idx == n_evaluations - 1:
                    agent.save(f"checkpoints/{agent}/final.pt")
        for ax, lineup in zip(axes[eval_idx], evaluation_lineups):
            rewards_per_tournament = evaluate(
                env, eval_size, lineup, desc=f"Evaluating after training session {eval_idx+1}")
            plot_results(rewards_per_tournament, lineup,
                         eval_size, window_size, ax)
    plt.tight_layout()
    save_fig(fig, directory="eval_logs")


def main():

    env = simulation.PokerTournament()

    dqn = DQNAgent(env, "dqn")
    if os.path.exists(f"checkpoints/{dqn}/final.pt"):
        dqn.load(f"checkpoints/{dqn}/final.pt")
        print(f"Loaded pretrained DQNAgent {dqn}")

    # dqn2 = DQNAgent(env, "dqn2")
    # if os.path.exists(f"checkpoints/{dqn2}/final.pt"):
    #     dqn2.load(f"checkpoints/{dqn2}/final.pt")
    #     print("Loaded pretrained DQNAgent dqn2")

    RANDOM_LINEUP = [dqn,RandomAllInFoldAgent(env), RandomAllInFoldAgent(env), RandomAllInFoldAgent(env)]
    ALL_IN_PAIR_LINEUP = [dqn, AllInPairAgent(env), AllInPairAgent(env), AllInPairAgent(env)]
    TWO_HIGH_LINEUP = [dqn, TwoHighAgent(env), TwoHighAgent(env), TwoHighAgent(env)]
    POOL = [RandomAllInFoldAgent, AllInPairAgent, TwoHighAgent, SuitedAgent]


    train_and_evaluate(env, N_total=100_000, learn_size=20_000, eval_size=1_000,
                       training_lineup=ALL_IN_PAIR_LINEUP,
                       evaluation_lineups=[ALL_IN_PAIR_LINEUP])
    



    env.close()
    
if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #s = io.StringIO()
    #pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(40)
    #print(s.getvalue())
