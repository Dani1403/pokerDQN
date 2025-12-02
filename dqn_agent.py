import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time 

from torch.utils.tensorboard import SummaryWriter



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
    'STATE_DIM': 10,
    'HIDDEN_DIM': 128,
    'GAMMA': 0.99,
    'LR': 5e-4,
    'BATCH_SIZE': 32,
    'BUFFER_SIZE': 50_000,
    'TARGET_SYNC': 500,
    'FREQ_TRAIN': 8,
    'EPS_START': 1.0,
    'EPS_END': 0.05,
    'EPS_DECAY': 5000,
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
        self.epsilon_start = H_PARAMS['EPS_START']
        self.epsilon_end = H_PARAMS['EPS_END']
        self.epsilon_decay = H_PARAMS['EPS_DECAY']
        self.epsilon = self.epsilon_start
        self.freq_train = H_PARAMS['FREQ_TRAIN']

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),     
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_actions)
        )

        self.target_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_actions)
        )
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        self.buffer = deque(maxlen=self.buffer_size)

        self.train_steps = 0


        self.writer = SummaryWriter(f"logs/run_{int(time.time())}")
        self.global_step = 0


    def forward(self, x):
        x = x.to(self.device)
        return self.net(x)

    def _preprocess_state(self, obs):
        low_rank, high_rank, suited = self.env._encode_hand(obs['hole_cards'])
        low_norm  = low_rank  / (self.env.n_ranks - 1)
        high_norm = high_rank / (self.env.n_ranks - 1)
        suited_norm = float(suited)

        stacks_bb = tuple(min(s // self.env.table.dealer.blinds[1], self.env.max_stack_bb - 1) for s in obs['stacks'])

        stack = stacks_bb[obs['action']]

        shortest = 1  # assume is shortest

        other_stacks = [stacks_bb[i] for i in range(
            len(stacks_bb)) if i != obs['action'] and obs['active'][i]]
        if other_stacks:
            shortest_other = min(other_stacks)
            if shortest_other < stack:
                shortest = 0

        player_idx = (obs['action'],)


        stack_norm = stack / (self.env.max_stack_bb - 1)

        shortest = 1
        other_stacks = [
            stacks_bb[i] for i in range(len(stacks_bb))
            if i != obs["action"] and obs["active"][i]
        ]

        other_stacks_norm = [s / (self.env.max_stack_bb - 1)
                             for s in other_stacks]

        if other_stacks:
            if min(other_stacks) < stack:
                shortest = 0

        shortest_norm = float(shortest)

        position_norm = obs["action"] / (self.env.num_players - 1)

        active_norm = sum(obs['active']) / self.env.num_players

        state = np.array([
            low_norm,
            high_norm,
            suited_norm,
            stack_norm,
            other_stacks_norm[0] if len(other_stacks_norm) > 0 else 0.0,
            other_stacks_norm[1] if len(other_stacks_norm) > 1 else 0.0,
            other_stacks_norm[2] if len(other_stacks_norm) > 2 else 0.0,
            active_norm,
            shortest_norm,
            position_norm
        ], dtype=np.float32)

        return state

    def act(self, state):

        state_t = torch.from_numpy(state).to(self.device).unsqueeze(0)

        if np.random.rand() < self.epsilon:
            action_idx = random.choice([0, 1])
        else:
            with torch.no_grad():
                q_values = self.forward(state_t)
                action_idx = q_values.argmax().item()

        return self.env.actions_to_env[action_idx]


    # Memorize experience in replay buffer
    def _remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def update_parameters(self, state, action, reward, next_state, done):

        if state is None:
            return

        action_idx = 0 if action == self.env.actions_to_env[0] else 1

        self._remember(state, action_idx, reward, next_state, done)
        self.global_step += 1

        #train every FREQ_TRAIN steps
        if self.global_step % self.freq_train != 0:
            return

        if len(self.buffer) < self.batch_size:
            return

        #sample from the replay buffer

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.stack(states)).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q values
        q_values = self.forward(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            # selection of max action from current network
            next_q_online = self.forward(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            # evaluation of next actions from target network
            next_q_target = self.target_net(
                next_states).gather(1, next_actions)
            target_q_values = rewards + self.gamma * \
                (1 - dones) * next_q_target
            
        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        # Logging
        # Log metrics
        if self.global_step % 100 == 0:
            self.writer.add_scalar("Loss/TD_Error", loss.item(), self.global_step)
            self.writer.add_scalar("Policy/Epsilon", self.epsilon, self.global_step)

        # Log average Q-values
        with torch.no_grad():
            avg_q = q_values.mean().item()
        self.writer.add_scalar("Q_values/avg_q", avg_q, self.global_step)

        self.global_step += 1


        self.train_steps += 1
        # Update target network
        if self.train_steps % self.target_sync == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # Decay epsilon
        if done and self.epsilon > self.epsilon_end:
            self.epsilon -= (self.epsilon_start - self.epsilon_end) / self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)



    def __str__(self):
        return self.__class__.__name__