import simconfig
from simconfig import *
import gymnasium as gym
from typing import Sequence, Tuple, Any, Union
import numpy as np
from collections import defaultdict

"""
Poker Tournament Simulation Wrapper for Clubs Gym Environment
This module wraps the Clubs Gym environment for simulating a poker tournament.
This environment simulates a poker tournament with configurable parameters such as
number of players, blind schedule, hands per level, and starting stack size.
It provides methods to reset the environment, step through the game, and render the game state.
Args : 
    players (int): Number of players in the tournament.
    blind_schedule (Sequence[int]): Schedule of blinds for the tournament.
    hands_per_level (int): Number of hands played per blind level.
    starting_stack_bb (int): Starting stack size in big blinds.
    prize_pool (Sequence[int]): Prize pool for the tournament
        format : [1st place, 2nd place, 3rd place, ...]
    render_mode (str | None): Rendering mode for visualizing the game state.
"""
class PokerTournament(gym.Env):

    metadata = {"render_modes": ["ascii", "human"]}

    def __init__(
        self,
        players: int = NUM_PLAYERS,
        blind_schedule: Sequence[int] = BLIND_SCHEDULE,
        hands_per_level: int = HANDS_PER_LEVEL,
        starting_stack: int = START_STACK,
        prize_pool: Sequence[int] = PRIZE_POOL,
        render_mode: Union[str, None] = "ascii",
    ):
        super().__init__()

        self.table = gym.make(GAME_ID, disable_env_checker=True).unwrapped

        self.observation_space = self.table.observation_space
        self.action_space = self.table.action_space

        self._blind_schedule = list(blind_schedule)
        self._hands_per_level = hands_per_level
        self._level = 0
        self._hands_played = 0
        self._prize_pool = prize_pool

        self.num_players = self.table.dealer.num_players

        self.max_stack = players * starting_stack

        bb = self.table.dealer.blinds[1]
        self.max_stack_bb = int(self.max_stack // bb) + 1 

        self.all_in = players * self.table.dealer.start_stack
        self.fold = -1

        self._render_mode = render_mode

        self._set_blinds()

        self._reward = [0] * players  # Rewards for each player
        self._bust_order = [None] * players  # Track the order of busts
        self.elimination_counter = 0

        self.actions = ['ALLIN', 'FOLD']              
        self.actions_to_env = {
            0: self.all_in,
            1: self.fold,
        }

        # --- Ranks normalization ---
        self.rank_min = min(CHAR_RANK_TO_INT_RANK.values())  
        self.rank_max = max(CHAR_RANK_TO_INT_RANK.values()) 
        self.n_ranks  = self.rank_max - self.rank_min + 1  

    """
    Reset the environment to its initial state.
    Args:
        seed (int | None): Random seed for reproducibility.
        options (dict | None): Additional options for resetting the environment.
    """
    def reset(
        self,
        *,
        seed: Union[int, None] = None,
        options: Union[dict, None] = None
    ) -> Tuple[Any, dict]:

        super().reset(seed=seed)

        #if we reset the whole tournament
        if options["reset_stacks"]:
            self._hands_played = 0
            self._level = 0

            self._reward = [0] * self.num_players  # Rewards for each player
            self._bust_order = [None] * self.num_players # Track the order of busts
            self.elimination_counter = 0

        self._set_blinds()

        obs = self.table.reset(reset_button=options["reset_button"], reset_stacks=options["reset_stacks"])
        info: dict = {}

        return obs, info

    """
    Step through the environment with the given action.
    Args:
        action (int): The action to take in the environment.
    Returns:
        Tuple[Any, float | list[float], bool, bool, dict]:
            - obs: The observation after taking the action.
            - rewards: The reward received after taking the action.
            - terminated: Whether the episode has terminated.
            - truncated: Whether the episode has been truncated.
            - info: Additional information about the step.
    """
    def step(self, action: int) -> Tuple[Any, Union[float, list[float]], bool, bool, dict]:

        obs, rewards, done, _ = self.table.step(action)
        # done is a list of booleans indicating if each player is done
        done = all(done)
        # if done is True, it means the hand is over
        if done:
            self._hands_played += 1
            if self._hands_played % self._hands_per_level == 0:
                self._level += 1
            self._set_blinds()

        if done:
            # update busted array
            busted = [i for i in range(self.num_players) \
                if self.table.dealer.stacks[i] == 0 and self._bust_order[i] is None]

            if busted:
                for i in busted:
                    self._bust_order[i] = self.elimination_counter
                self.elimination_counter += 1

        # the game is over when all but one player has no chips left at the end of the hand
        game_over = done and (self.table.dealer.stacks.count(0) == self.num_players - 1)

        info = {}

        if game_over:
            #assign prize pool based on bust order
            winner = self.table.dealer.stacks.index(
                max(self.table.dealer.stacks))
            if self._bust_order[winner] is None:
                self._bust_order[winner] = self.elimination_counter
            rewards = self._assign_rewards(self._bust_order, self._prize_pool)
            info['hand_done'] = True
            info['post_payout_stacks'] = list(self.table.dealer.stacks)
            return obs, rewards, True, False, info

        elif done: #the game is not over
            # capture post-payout stacks BEFORE reset
            info['hand_done'] = True
            info['post_payout_stacks'] = list(self.table.dealer.stacks)
            # use fresh obs from reset
            obs, _ = self.reset(options={"reset_button": False, "reset_stacks": False})

        return obs, [0] * self.table.dealer.num_players, game_over, False, info

    """
    Set the blinds for the current level based on the blind schedule.
    This method updates the blinds according to the predefined blind schedule.
    """
    def _set_blinds(self):
        sb = self._blind_schedule[min(self._level, len(self._blind_schedule) - 1)]
        bb = sb * 2
        self.table.dealer.blinds = [sb, bb] + \
            [0] * (self.num_players - 2)
        self.table.dealer.big_blind = bb
        self.max_stack_bb = int(self.max_stack // bb) + 1


    def _get_num_active_players(self):
        return sum(stack > 0 for stack in self.table.dealer.stacks)

    def _assign_rewards(self, bust_order, prize_pool):
        reward = [0] * len(bust_order)
        order_to_players = defaultdict(list)

        #group players by bust order
        for i, order in enumerate(bust_order):
            order_to_players[order].append(i)

        # sort the orders in descending order, from winner to losers
        sorted_orders = sorted(order_to_players.keys(), reverse=True)

        # distribute the prize pool based on the bust order
        prize_index = 0
        for order in sorted_orders:
            players = order_to_players[order]
            num = len(players)

            #stop if we are out of prize
            if prize_index >= len(prize_pool):
                break

            #average the prize for the players in this order, in case of ties
            end = min(prize_index + num, len(prize_pool))
            prize_slice = prize_pool[prize_index:end]
            avg = sum(prize_slice) / len(players)

            # assign the average prize to each player in this order
            for p in players:
                reward[p] = avg

            prize_index += num

        return reward




    def _encode_hand(self, hand):
        """Encodes the hand as (low_rank_idx, high_rank_idx, suited_idx) with ranks 0..n_ranks-1."""
        c1 = CHAR_RANK_TO_INT_RANK[hand[0].rank] - self.rank_min
        c2 = CHAR_RANK_TO_INT_RANK[hand[1].rank] - self.rank_min
        low, high = (c1, c2) if c1 <= c2 else (c2, c1)
        suited = 1 if (hand[0].suit == hand[1].suit) else 0
        return (low, high, suited)


    """
    Render the current state of the environment.
    """
    def render(self):
        if self._render_mode:
            self.table.render(mode=self._render_mode)