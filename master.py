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
from multiprocessing import Process, Queue
from replay_buffer import ReplayBuffer
import torch
from worker import worker_process



def train_parallel(n_workers=6, total_steps=200_000):

    from multiprocessing import Process, Queue, Pipe
    from replay_buffer import ReplayBuffer
    from dqn_agent import DQNAgent

    env = simulation.PokerTournament()
    dqn = DQNAgent(env, "DQN-MP")

    queue = Queue(maxsize=20000)
    buffer = ReplayBuffer(capacity=2_000_000)

    params = {
        "state_dim": dqn.state_dim,
        "hidden_dim": dqn.hidden_dim,
        "epsilon": dqn.epsilon_start,
    }

    # Launch workers
    workers = []
    pipes = []
    for wid in range(n_workers):
        parent_conn, child_conn = Pipe()
        p = Process(target=worker_process,
                    args=(wid, queue, params, child_conn))
        p.daemon = True
        p.start()
        pipes.append(parent_conn)
        workers.append(p)

    # Send initial weights
    weights = dqn.net.state_dict()
    for conn in pipes:
        conn.send(weights)

    print("Workers started.")

    for step in range(total_steps):

        # get transition
        s, a, r, s2, done = queue.get()
        buffer.push(s, a, r, s2, done)

        # train periodically
        if len(buffer) > dqn.batch_size and step % 32 == 0:
            batch = buffer.sample(dqn.batch_size)
            dqn.train_batch(*batch)

        # update workers every 5000 steps
        if step % 5000 == 0:
            weights = dqn.net.state_dict()
            for conn in pipes:
                conn.send(weights)

        if step % 2000 == 0:
            print(f"Step {step}, buffer size {len(buffer)}")


