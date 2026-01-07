from dqn_agent import DQNAgent
from poker_agents import *
import simulation
import multiprocessing as mp
import torch
from run_tournaments import run_tournament, run_n_tournaments
from eval import evaluate, plot_results, save_fig, parallel_eval
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from poker_dqn import Poker_DQN

""" must provide a list of agents forming the base of the training pool """
def training_table(env, agents, pool):
    for _ in range(env.num_players - len(agents)) :
        Opponent = random.choice(pool)
        agents.append(Opponent(env))
    return agents


def train(env, n_tournaments, lineup, desc):
    run_n_tournaments(env, n_tournaments, evaluate=False, fixed_lineup=lineup, desc=desc)


def train_and_save(env, N_total, learn_size, training_lineup, checkpoint_root="checkpoints"):
    n_evaluations = N_total // learn_size
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    agent_ckpt_dirs = {}
    for agent in training_lineup:
        if hasattr(agent, "save"):
            agent_name = str(agent)
            ckpt_dir = os.path.join(checkpoint_root, f"{agent_name}_{timestamp}")
            os.makedirs(ckpt_dir, exist_ok=True)
            agent_ckpt_dirs[agent] = ckpt_dir

    #training loop 
    for eval_idx in range(n_evaluations):
        print(f"[TRAINING] Starting training iteration {eval_idx + 1}/{n_evaluations}")
        train(env, learn_size, training_lineup,
              desc=f"Training Iteration {eval_idx + 1}/{n_evaluations}")
        # save checkpoints
        for agent, ckpt_dir in agent_ckpt_dirs.items():
            agent.save(os.path.join(ckpt_dir, f"iter_{eval_idx + 1}_{(eval_idx+1) * learn_size}.pt"))

    # save final model after last iteration
    for agent, ckpt_dir in agent_ckpt_dirs.items():
        agent.save(os.path.join(ckpt_dir, f"final.pt"))

    print("[TRAINING] Training completed and checkpoints saved.")
    ckpt_dirs = {str(agent): dir for agent, dir in agent_ckpt_dirs.items()}
    return ckpt_dirs

"""
Multiprocessing training setup
"""
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

def learner(dqn, queue, stop_event, max_transitions, save_every, model_path):
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

def train_and_evaluate_mp():
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

    learner(dqn, queue, stop_event, max_transitions=max_transitions,
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