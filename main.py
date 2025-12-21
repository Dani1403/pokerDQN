import time
from matplotlib import axes
import torch
from torch import t
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
import multiprocessing as mp

def run_tournament(env, agents, evaluate=False, collect_transitions=False):

    transitions = []

    obs, _ = env.reset(options={"reset_button": True, "reset_stacks": True})
    prev_stacks = obs['stacks'].copy()
    done = False

    while not done:

        player_idx = env.table.dealer.action
        curr_agent = agents[player_idx]

        #acton selection
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

        # Training step or collect transition
        if isinstance(curr_agent, DQNAgent):
            if collect_transitions:
                transitions.append((state, action, reward[player_idx], next_state, done))

            elif not evaluate:
                curr_agent.update_parameters(
                    state,
                    action,
                    reward[player_idx],
                    next_state,
                    done
                )

        prev_stacks = new_stacks
        obs = next_obs
    if collect_transitions:
        return transitions
    else:
        return reward

def worker(worker_id, queue, model_path, opponents, stop_event, sync_every, worker_epsilon):
    queue.cancel_join_thread()
    env = simulation.PokerTournament()
    dqn = DQNAgent(env, f"worker_{worker_id}", device="cpu", enable_tb=False)
    dqn.load(model_path, map_location="cpu")
    dqn.epsilon = worker_epsilon
    dqn.net.eval()
    local_steps = 0
    lineup = [dqn] + [op(env) for op in opponents]

    while not stop_event.is_set():
        transitions = run_tournament(env, lineup, evaluate=False, collect_transitions=True)
        for t in transitions:
            try: 
                queue.put(t, timeout=0.1)
            except:
                pass
            local_steps += 1

            if local_steps % sync_every == 0:
                try:
                    dqn.load(model_path, map_location="cpu")
                    dqn.epsilon = worker_epsilon
                except Exception as e:
                    pass
    env.close()
    try:
        dqn.writer.close()
    except:
        pass
    print("worker exit")

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


def learner(dqn, queue, env, stop_event, max_transitions, save_every, model_path):
    processed = 0
    while processed < max_transitions and not stop_event.is_set():
        try:
            state, action, reward, next_state, done = queue.get(timeout=1)
        except Exception:
            continue
        dqn.update_parameters(state, action, reward, next_state, done)
        processed += 1
        if processed % save_every == 0:
            it = processed // save_every
            dqn.save(model_path)
            dqn.save(f"{model_path[:-3]}_{it}.pt")
            print(f"[LEARNER] Saved model at {processed} transitions")

    dqn.save(model_path)
    stop_event.set()


""" must provide a fixed lineup """
def run_n_tournaments(env, n_tournaments, evaluate=False, fixed_lineup=None, desc=None, show_tqdm=True):

    rewards_per_tournament = []
    saved_epsilons = {}

    if evaluate:
        for agent in fixed_lineup:
            if hasattr(agent, "epsilon"):
                saved_epsilons[agent] = agent.epsilon
                agent.epsilon = 0.0

    iterator = range(n_tournaments)
    if show_tqdm:
        iterator = tqdm(iterator, ascii=True, ncols=80, desc=desc)

    for _ in iterator:

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


def evaluate(env, n_tournaments, lineup, desc, show_tqdm=True):
    rewards_per_tournament = run_n_tournaments(
        env, n_tournaments, evaluate=True, fixed_lineup=lineup, desc=desc, show_tqdm=show_tqdm)
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
    dqn2 = DQNAgent(env, "dqn2")
    if os.path.exists(f"checkpoints/{dqn2}/final.pt"):
        dqn2.load(f"checkpoints/{dqn2}/final.pt")
        print("Loaded pretrained DQNAgent dqn2")
    RANDOM_LINEUP = [dqn,RandomAllInFoldAgent(env), RandomAllInFoldAgent(env), RandomAllInFoldAgent(env)]
    ALL_IN_PAIR_LINEUP = [dqn, AllInPairAgent(env), AllInPairAgent(env), AllInPairAgent(env)]
    TWO_HIGH_LINEUP = [dqn, TwoHighAgent(env), TwoHighAgent(env), TwoHighAgent(env)]
    POOL = [RandomAllInFoldAgent, AllInPairAgent, TwoHighAgent, SuitedAgent]

    train_and_evaluate(env, N_total=100_000, learn_size=20_000, eval_size=1_000,
                       training_lineup=ALL_IN_PAIR_LINEUP,
                       evaluation_lineups=[ALL_IN_PAIR_LINEUP])
    env.close()


def main_mp():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp.set_start_method('spawn', force=True)
    env = simulation.PokerTournament()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dqn = DQNAgent(env, "dqn_mp", device=device, enable_tb=True)
    model_path = f"checkpoints/dqn_{timestamp}/longrun.pt"
    if os.path.exists(model_path):
        dqn.load(model_path, map_location="cpu")
        print("Loaded pretrained DQNAgent dqn_mp")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    dqn.save(model_path)
    queue = mp.Queue(maxsize=200_000)
    stop_event = mp.Event()
    opponents = [AllInPairAgent, AllInPairAgent, AllInPairAgent]
    num_workers = 12
    workers = []
    eps_min = 0.05
    eps_max = 0.9
    if num_workers == 1:
        workers_epsilons = [0.5]
    else :
        workers_epsilons = [eps_min + i * ((eps_max - eps_min) / (num_workers - 1))
                        for i in range(num_workers)]
    for wid in range(num_workers):
        p = mp.Process(target=worker, args=(wid, queue, model_path, opponents, stop_event, 5_000, workers_epsilons[wid]))
        p.start()
        workers.append(p)

    max_transitions = 25_000_000
    save_every = 500_000

    learner(dqn, queue, env, stop_event, max_transitions=max_transitions,
            save_every=save_every, model_path=model_path)

    stop_event.set()
    queue.close()
    queue.join_thread()
    for p in workers:
        p.join(timeout=10)
        print("joined?", not p.is_alive(), "exitcode:", p.exitcode)   
    env.close()

    # Evaluate the final model and the checkpoints
    num_checkpoints = max_transitions // save_every
    EVAL_EVERY = 5
    eval_dir = f"eval_logs/dqn_{timestamp}"
    os.makedirs(eval_dir, exist_ok=True)
    for it in range(num_checkpoints):
        if (it + 1) % EVAL_EVERY != 0:
            continue
        iter_model_path = f"{model_path[:-3]}_{it+1}.pt"
        if os.path.exists(iter_model_path):
            print(f"[MAIN EVAL] Evaluating checkpoint at iteration {it+1}")
            parallel_eval(iter_model_path, name=f"eval_iter_{it+1}.png", eval_dir=eval_dir)
    if os.path.exists(model_path):
        parallel_eval(model_path, name="evaluation_final.png", eval_dir=eval_dir)

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


if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    #main()
    #main_mp()
    eval_checkpoint_dir("checkpoints/dqn/20251217_205057_434015") #continuing of best model
    eval_checkpoint_dir("checkpoints/dqn/20251218_085359_791915") #best model so far at the end
    #pr.disable()
    #s = io.StringIO()
    #pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(40)
    #print(s.getvalue())
