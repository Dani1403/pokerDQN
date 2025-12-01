import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


import numpy as np
import random

"""Example of an observation dictionary:
        >>> dealer.step(0)
        ... ({'action': 0, which player is acting (0 = player 1, 1 = player 2, ...),
        ...  'active': [True, True],  which player is in the hand or not (True = in, False = out or busted)
        ...  'button': 1, which player is the dealer (0 = player 1, 1 = player 2, ...),
        ...  'call': 0, the amount needed to call the current bet,
        ...  'community_cards': [], the community cards on the table,
        ...  'hole_cards': [[Card (139879188163600): A (symb heart) ], [Card (139879188163504): A (symb spades)]], the hole cards of the player,
        ...  'max_raise': 2, the maximum amount that can be raised,
        ...  'min_raise': 2, the minimum amount that can be raised,
        ...  'pot': 2, the total amount of chips in the pot,
        ...  'stacks': [9, 9], the current stacks of each player,
        ...  'street_commits': [0, 0]}


        stacks in blinds (conversion needed)
        prize pool 


        start with small state space with q learning vs all random
"""

"""
Example Hyper parameter dictionary for the DQN agent :
    hparams = {
        GAMMA        = 0.99     discount factor
        LR           = 5e-4     learning rate
        BATCH_SIZE   = 128      
        BUFFER_SIZE  = 50_000
        TARGET_SYNC  = 500        
        EPS_START    = 1.0   
        EPS_END      = 0.05
        EPS_DECAY    = 5_000      
    }
"""

H_PARAMS = {
    'N_ACTIONS': 2,
    'STATE_DIM': 3,
    'HIDDEN_DIM': 128,
    'GAMMA': 0.99,
    'LR': 5e-4,
    'BATCH_SIZE': 128,
    'BUFFER_SIZE': 50_000,
    'TARGET_SYNC': 500,
    'EPS_START': 1.0,
    'EPS_END': 0.05,
    'EPS_DECAY': 5_000,
}


class DQNAgent(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.env = env

        self.n_actions = H_PARAMS['N_ACTIONS']
        self.state_dim = H_PARAMS['STATE_DIM']
        self.hidden_dim = H_PARAMS['HIDDEN_DIM']
        self.gamma = H_PARAMS['GAMMA']
        self.lr = H_PARAMS['LR']
        self.batch_size = H_PARAMS['BATCH_SIZE']
        self.buffer_size = H_PARAMS['BUFFER_SIZE']
        self.target_sync = H_PARAMS['TARGET_SYNC']
        self.eps_start = H_PARAMS['EPS_START']
        self.eps_end = H_PARAMS['EPS_END']
        self.eps_decay = H_PARAMS['EPS_DECAY']

        self.epsilon = self.eps_start

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),     
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_actions)
        )

    def forward(self, x):
        return self.net(x)

    def _preprocess_state(self, obs):
        # For now just use hand encoding as state
        hand_encoding = self.env._encode_hand(obs['hole_cards'])
        return np.array(hand_encoding, dtype=np.float32)

    def act(self, obs):
        
        state = self._preprocess_state(obs)
        if np.random.rand() < self.epsilon:
            action_idx = random.choice([0, 1])
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.forward(state)
        action_idx = q_values.argmax().item()   
        return self.env.actions_to_env[action_idx]

    def update_parameters(self, obs, action, reward, next_obs, done):
        pass
    


    def __str__(self):
        return self.__class__.__name__