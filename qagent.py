import numpy as np
import random
from clubs.poker.card import CHAR_RANK_TO_INT_RANK

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

        start with small state space with q learning vs all random
"""

hparams = {
    'EPS_START': 1.0,
    'EPS_END': 0.05,
    'EPS_DECAY': 0.9995,
    'GAMMA': 0.99,
    'LR': 1e-3,
    'BINS': [0,3,5,10,20,40,999],
}

class QAgent:
    def __init__(self, env):
        self.env = env
        self.hparams = hparams
        self.epsilon = hparams['EPS_START']
        self.epsilon_end = hparams['EPS_END']
        self.epsilon_decay = hparams['EPS_DECAY']
        self.gamma = hparams['GAMMA']
        self.lr = hparams['LR']
        self.bins = hparams['BINS']

        # --- Ranks normalization ---
        self.rank_min = min(CHAR_RANK_TO_INT_RANK.values())  
        self.rank_max = max(CHAR_RANK_TO_INT_RANK.values()) 
        self.n_ranks  = self.rank_max - self.rank_min + 1    

        # --- Big blind & stack caps ---
        bb = self.env.table.dealer.blinds[1]
        self.bb = bb
        start_stack = getattr(self.env, 'start_stack', getattr(self.env, 'max_stack', None))
        self.max_stack_bb = int(start_stack // bb) + 1  

        # --- Action mapping (ensure consistency) ---
        self.ACTIONS = ['ALLIN', 'FOLD']               # index 0 -> ALLIN, 1 -> FOLD
        self.ACTION_TO_ENV = {
            0: self.env.all_in,
            1: self.env.fold,
        }

        # --- Q-table shape ---
        shape = [env.num_players,  # position of player
                 self.n_ranks,     # low card rank
                 self.n_ranks,     # high card rank
                 2,                # suited or not
                 len(self.bins)-1,  # bucketized stack
                 2,                # bool : 0 - player not shortest stack / 1 - player is shortest stack
                 env.num_players,  # num active players -1 :
                 len(self.ACTIONS)]
        self.q_table = np.zeros(shape, dtype=np.float32)


    def _encode_hand(self, hand):
        """Encodes the hand as (low_rank_idx, high_rank_idx, suited_idx) with ranks 0..n_ranks-1."""
        c1 = CHAR_RANK_TO_INT_RANK[hand[0].rank] - self.rank_min
        c2 = CHAR_RANK_TO_INT_RANK[hand[1].rank] - self.rank_min
        low, high = (c1, c2) if c1 <= c2 else (c2, c1)
        suited = 1 if (hand[0].suit == hand[1].suit) else 0
        return (low, high, suited)

    def _preprocess_state(self, obs):
        """Preprocess the observation to get the state representation."""
        # Normalize stacks to BB and clip to table shape to avoid OOB
        stacks_bb = tuple(min(s // self.bb, self.max_stack_bb - 1) for s in obs['stacks'])

        stack = stacks_bb[obs['action']]
        
        bucket = np.digitize(stack, self.bins) - 1

        bucket = max(0, min(bucket, len(self.bins) - 2))

        shortest = 1  # assume is shortest

        other_stacks = [stacks_bb[i] for i in range(
            len(stacks_bb)) if i != obs['action'] and obs['active'][i]]
        if other_stacks:
            shortest_other = min(other_stacks)
            if shortest_other < stack:
                shortest = 0

        num_active = sum(obs['active']) - 1

        player_idx = (obs['action'],)
        hand = self._encode_hand(obs['hole_cards'])
        return player_idx + hand + (bucket, shortest, num_active)

    def act(self, obs):
        """Select an action based on the current observation."""
        state = self._preprocess_state(obs)
        if np.random.rand() < self.epsilon:
            action_idx = random.choice([0, 1])
        else:
            q_vals = self.q_table[state]  
            action_idx = int(np.argmax(q_vals))
        return self.ACTION_TO_ENV[action_idx]

    def update_parameters(self, obs, action, reward, next_obs=None, done=False):
        if action == self.env.all_in:
            action_idx = 0
        else:
            action_idx = 1

        state = self._preprocess_state(obs)

        if next_obs is None or next_obs.get('action', -1) == -1:
            next_max = 0.0
        else:
            next_state = self._preprocess_state(next_obs)
            next_max = float(np.max(self.q_table[next_state]))

        old_value = self.q_table[state + (action_idx,)]
        target = reward + self.gamma * next_max
        new_value = old_value + self.lr * (target - old_value)
        self.q_table[state + (action_idx,)] = new_value

        if done and self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def __str__(self):
        return self.__class__.__name__
