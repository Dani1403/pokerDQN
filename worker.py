import os
import gymnasium as gym
import simulation
from poker_agents import *
from qagent import QAgent
from dqn_agent import DQNAgent
import numpy as np

import torch
import torch.nn as nn

import simulation
from multiprocessing import Queue

class WorkerPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_actions, device):
        super().__init__()
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        ).to(device)

    def load_weights(self, weights):
        self.load_state_dict(weights)

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice([0, 1])

        t = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.net(t)
        return q.argmax().item()


def worker_process(worker_id, queue, params, weights_pipe):
    # params is a dict containing dims etc.
    env = simulation.PokerTournament()
    device = torch.device("cpu")

    policy = WorkerPolicy(
        state_dim=params["state_dim"],
        hidden_dim=params["hidden_dim"],
        n_actions=2,
        device=device
    )

    # receive initial weights
    policy.load_weights(weights_pipe.recv())

    obs, _ = env.reset(options={"reset_button": True, "reset_stacks": True})
    prev_stacks = obs["stacks"]

    epsilon = params["epsilon"]

    while True:
        state = env._encode_hand(obs["hole_cards"])  # or DQNAgent preprocess

        action_idx = policy.act(state, epsilon)
        action_env = env.actions_to_env[action_idx]

        next_obs, reward, done, _, _ = env.step(action_env)

        # Compute shaped rewards
        new_stacks = next_obs["stacks"]
        diffs = np.array(new_stacks) - np.array(prev_stacks)
        diffs = np.tanh(diffs / 5.0)

        next_state = None if done else env._encode_hand(next_obs["hole_cards"])

        # send sample for player 0
        queue.put((state, action_idx, diffs[0], next_state, done))

        prev_stacks = new_stacks
        obs = next_obs

        # receive updated weights if master sends them
        if weights_pipe.poll():
            new_weights = weights_pipe.recv()
            policy.load_weights(new_weights)
