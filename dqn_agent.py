import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np




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

OBS_DIM = 11  
N_ACTIONS = 2 #all in - fold

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
            nn.Linear(128, N_ACTIONS)
        )
    def forward(self, x):
        return self.net(x)