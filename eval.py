import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing as mp
import simulation
from dqn_agent import DQNAgent
from poker_agents import *
from run_tournaments import run_n_tournaments


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


def worker_eval(worker_id, model_path, opponents, n_tournaments, return_dict):
    env = simulation.PokerTournament()
    dqn = DQNAgent(env, f"eval_worker_{worker_id}",
                   device="cpu", enable_tb=False)
    dqn.load(model_path, map_location="cpu")
    dqn.epsilon = 0.0
    dqn.net.eval()
    lineup = [dqn] + [op(env) for op in opponents]
    rewards_per_tournament = evaluate(
        env, n_tournaments, lineup, desc=None, show_tqdm=False)
    return_dict[worker_id] = rewards_per_tournament
    env.close()
    print("eval worker exit")

def eval_lineup_parallel(name, model_path, opponents, n_workers, n_tournaments_per_worker, return_dict):
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    local_return = manager.dict()
    workers = []
    for wid in range(n_workers):
        p = mp.Process(
            target=worker_eval,
            args=(wid, model_path, opponents,
                  n_tournaments_per_worker, local_return)
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
    all_rewards = []
    for wid in range(n_workers):
        all_rewards.extend(local_return[wid])
    return_dict[name] = all_rewards


def evaluate(env, n_tournaments, lineup, desc, show_tqdm=True):
    rewards_per_tournament = run_n_tournaments(
        env, n_tournaments, evaluate=True, fixed_lineup=lineup, desc=desc, show_tqdm=show_tqdm)
    return rewards_per_tournament


def parallel_eval(model_path, fig_name, eval_dir):
    print("[MAIN EVAL] Starting evaluation of iteration")
    lineups = {
        "AllInPair": [AllInPairAgent]*3,
        "Random":    [RandomAllInFoldAgent]*3,
        "TwoHigh":   [TwoHighAgent]*3,
        "Suited":    [SuitedAgent]*3,
    }
    n_workers_per_lineup = 2
    n_tournaments_per_worker = 1000
    manager = mp.Manager()
    final_results = manager.dict()
    lineup_processes = []
    # ---- start one process per lineup ----
    for lineup_name, ops in lineups.items():
        p = mp.Process(
            target=eval_lineup_parallel,
            args=(
                lineup_name,
                model_path,
                ops,
                n_workers_per_lineup,
                n_tournaments_per_worker,
                final_results
            )
        )
        p.start()
        lineup_processes.append(p)
    # ---- wait for all lineups to finish ----
    for p in lineup_processes:
        p.join()
    # ---- plotting (single process) ----
    env = simulation.PokerTournament()
    dqn = DQNAgent(env, "eval", device="cpu", enable_tb=False)
    dqn.load(model_path, map_location="cpu")
    n_lineups = len(final_results)
    fig, axes = plt.subplots(n_lineups, 1, figsize=(16, 5*n_lineups), sharex=False)
    if n_lineups == 1:
        axes = [axes]
    for ax, (name, rewards) in zip(axes, final_results.items()):
        lineup = [dqn] + [op(env) for op in lineups[name]]

        plot_results(
            rewards,
            lineup,
            n_tournaments=len(rewards),
            window_size=200,
            ax=ax
        )
        ax.set_title(f"Evaluation against {name} lineup")
    fig.suptitle("Evaluation of DQNAgent against various lineups", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.97])
    save_fig(fig, name=fig_name, directory=eval_dir)
    env.close()


def eval_checkpoint_dir(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(
        checkpoint_dir) if f.endswith('.pt')]
    checkpoint_files.sort()
    for ckpt in checkpoint_files:
        model_path = os.path.join(checkpoint_dir, ckpt)
        print(f"[EVAL] Evaluating {model_path}")
        parallel_eval(model_path, fig_name=f"eval_{ckpt[:-3]}.png",
                      eval_dir=f"eval_logs/{os.path.basename(checkpoint_dir)}")
