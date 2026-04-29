import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing as mp
from clubs import poker
import simulation
from dqn_agent import DQNAgent
from poker_agents import *
from run_tournaments import run_n_tournaments
import os
from poker_dqn import Poker_DQN

def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

def cumulative_average(x):
    x = np.array(x)
    return np.cumsum(x) / np.arange(1, len(x) + 1)

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


def plot_cumulative_results(rewards_per_tournament, agents, n_tournaments, ax):
    reward_per_agent = np.array(rewards_per_tournament).T
    placement_summary = placements(rewards_per_tournament)

    final_avg_rewards = []

    for i, rewards in enumerate(reward_per_agent):
        cum_avg = cumulative_average(rewards)
        x_values = range(len(cum_avg))

        final_avg_rewards.append(cum_avg[-1])

        ax.plot(x_values, cum_avg, linewidth=2, label=f"Player {i + 1}")

    ax.set_title("Cumulative Average Reward per Tournament")
    ax.set_xlabel("Tournament")
    ax.set_ylabel("Cumulative Average Reward")

    x_ticks_spacing = max(1, n_tournaments // 10)
    ax.set_xticks(range(0, n_tournaments, x_ticks_spacing))

    n_agents = min(len(agents), len(final_avg_rewards), len(placement_summary))

    ax.legend([
        f"Player {i + 1} : {agents[i]}\n"
        f"Final avg reward: {final_avg_rewards[i]:.2f}\n"
        f"Placements: {placement_summary[i]}"
        for i in range(n_agents)
    ])

    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.4)

"""
Multiprocessing evaluation setup
"""


"""
agents = [
    {'type' : AgentClass, 'model_path' : 'path/to/model.pt' or None, 'name': name of agent},
    ...
]
"""
def worker_eval(worker_id, agents, n_tournaments, return_dict):
    print("[WORKER EVAL] Starting eval worker", worker_id)
    env = simulation.PokerTournament()
    lineup = []
    for agent in agents:
        agent_type = agent['type']
        if isinstance(agent_type, type) and agent_type.__name__=="Poker_DQN":
            poker_dqn = agent_type(
                env, name=f"eval_worker_{worker_id}_{agent['name']}", device="cpu", enable_tb=False)
            poker_dqn.state_dqn.net.eval()
            poker_dqn.state_dqn.epsilon = 0.0
            if agent.get('model_path') is not None:
                poker_dqn.load(agent['model_path'], map_location="cpu")
            lineup.append(poker_dqn)
        elif agent_type.__class__.__name__ == "Poker_DQN":
            poker_dqn = agent_type
            poker_dqn.state_dqn.net.eval()
            lineup.append(poker_dqn)
        elif isinstance(agent_type, type) and issubclass(agent_type, DQNAgent):
            dqn = agent_type(
                env, f"eval_worker_{worker_id}_{agent['name']}", device="cpu", enable_tb=False)
            dqn.net.eval()
            dqn.epsilon = 0.0
            if agent.get('model_path') is not None:
                dqn.load(agent['model_path'], map_location="cpu")
            lineup.append(dqn)
        elif isinstance(agent_type, DQNAgent):
            dqn = agent_type
            dqn.net.eval()
            lineup.append(dqn)

        else:
            try:
                lineup.append(agent['type'](env))
            except TypeError as e:
                raise RuntimeError(
                    f"Error initializing agent {agent['name']}: {e}") from e

    rewards_per_tournament = evaluate(
        env, n_tournaments, lineup, desc=None, show_tqdm=False)
    return_dict[worker_id] = rewards_per_tournament
    env.close()
    print("[WORKER EVAL] Finished eval worker", worker_id)

def eval_lineup_parallel(name, agents, n_workers, n_tournaments_per_worker, return_dict):
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    local_return = manager.dict()
    workers = []
    for wid in range(n_workers):
        p = mp.Process(
            target=worker_eval,
            args=(wid, agents,
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

def evaluate(env, n_tournaments, lineup, desc, show_tqdm=False):
    rewards_per_tournament = run_n_tournaments(
        env, n_tournaments, evaluate=True, fixed_lineup=lineup, desc=desc, show_tqdm=show_tqdm)
    return rewards_per_tournament

"""
lineups : dict 
{ lineup_name : [agent_spec, agent_spec, ...], ...
"""
def parallel_eval(lineups, eval_name, eval_dir, n_workers_per_lineup=2, n_tournaments_per_worker=1000):
    print(f"[PARALLEL EVAL] Starting eval {eval_name} ({len(lineups)} lineups)")
    manager = mp.Manager()
    final_results = manager.dict()
    lineup_processes = []
    # ---- start one process per lineup ----
    for lineup_name, agents in lineups.items():
        p = mp.Process(
            target=eval_lineup_parallel,
            args=(
                lineup_name,
                agents,
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
    # os.makedirs(eval_dir, exist_ok=True)
    # n_lineups = len(lineups)
    # fig, axes = plt.subplots(n_lineups,1,figsize=(16, 5 * n_lineups),sharex=False)
    # if n_lineups == 1:
    #     axes = [axes]
    # for ax, (lineup_name, rewards) in zip(axes, final_results.items()):
    #     agent_names = [a["name"] for a in lineups[lineup_name]]
    #     plot_cumulative_results(
    #         rewards_per_tournament=rewards,
    #         agents=agent_names,
    #         n_tournaments=len(rewards),
    #         ax=ax,
    #     )
    #     ax.set_title(f"Evaluation against {lineup_name} lineup")
    # fig.suptitle(
    #     f"Evaluation {eval_name}",
    #     fontsize=16,
    # )
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    # save_fig(fig, name=f"{eval_name}.png", directory=eval_dir)

    return dict(final_results)


def eval_checkpoint_dir(checkpoint_dirs,
                        n_workers_per_lineup=2,
                        n_tournaments_per_worker=1000):

    first_dir = next(iter(checkpoint_dirs.values()))
    run_id = os.path.basename(first_dir)
    eval_dir = os.path.join("eval_logs", run_id)
    os.makedirs(eval_dir, exist_ok=True)

    # ---------- helper: extract training step ----------
    def extract_step(name):
        if "final" in name:
            return float('inf')  # ensure final is last
        parts = name.replace(".pt", "").split("_")
        return int(parts[-1])  # last part = step number

    # ---------- collect checkpoints ----------
    ckpts_per_agent = {}
    for agent_name, ckpt_dir in checkpoint_dirs.items():
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        ckpts = sorted(ckpts, key=extract_step)
        ckpts_per_agent[agent_name] = ckpts

    common_ckpts = set.intersection(*(set(v) for v in ckpts_per_agent.values()))
    common_ckpts = sorted(common_ckpts, key=extract_step)

    print(f"[EVAL CHECKPOINTS] Found {len(common_ckpts)} checkpoints")

    # ---------- storage ----------
    results_over_time = {agent_name: [] for agent_name in checkpoint_dirs.keys()}
    final_rewards = None

    # ---------- evaluation loop ----------
    for ckpt in common_ckpts:
        print(f"[EVAL CHECKPOINTS] Evaluating {ckpt}")

        agents = []
        for agent_name, ckpt_dir in checkpoint_dirs.items():
            agents.append({
                'type': Poker_DQN,
                'model_path': os.path.join(ckpt_dir, ckpt),
                'name': agent_name
            })

        lineups = {"self_play": agents}

        results = parallel_eval(
            lineups=lineups,
            eval_name=ckpt,
            eval_dir=eval_dir,
            n_workers_per_lineup=n_workers_per_lineup,
            n_tournaments_per_worker=n_tournaments_per_worker
        )

        rewards = list(results.values())[0]
        final_rewards = rewards  # keep last checkpoint

        reward_per_agent = np.array(rewards).T
        avg_rewards = reward_per_agent.mean(axis=1)

        for i, agent_name in enumerate(checkpoint_dirs.keys()):
            results_over_time[agent_name].append(avg_rewards[i])

    # =====================================
    # FINAL PLOT
    # =====================================

    fig, ax = plt.subplots(figsize=(12, 7))

    agent_names = list(results_over_time.keys())

    # ---- x-axis = actual training steps ----
    x_values = [extract_step(c) / 1e6 for c in common_ckpts]
    for agent_name in agent_names:
        values = results_over_time[agent_name]
        cum_avg = cumulative_average(values)


        ax.plot(x_values, cum_avg, label=f"{agent_name} ({cum_avg[-1]:.3f})", linewidth=2)

    ax.set_title("Cumulative Performance over Training")
    ax.set_xlabel("Training Steps (millions)")
    ax.set_ylabel("Cumulative Average Reward")
    ax.legend(loc="best")
    ax.grid(True)

    

    save_fig(fig, name="training_progression_2500.png", directory=eval_dir)

