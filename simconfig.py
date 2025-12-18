import gymnasium as gym
import sys 
sys.modules["gym"] = gym #fix gymnasium-gym mismatches in registers
from clubs_gym import envs
from poker_agents import *
from qagent import QAgent

GAME_ID = "Poker-v0"


#PARAMETERS FOR THE TOURNAMENT SIMULATION

NUM_PLAYERS = 4

PRIZE_POOL = (150, 50, -50, -150)

HANDS_PER_LEVEL = 30

#in small blind
BLIND_SCHEDULE = (1, 2, 4, 6, 8, 12, 16, 24, 48) 

START_STACK = 50

START_BLINDS = [1, 2] + [0] * (NUM_PLAYERS - 2)



#configuration of the simulation DO NOT TOUCH
CONFIG = {
        "num_players": NUM_PLAYERS,
        "num_streets": 2,
        "blinds":      START_BLINDS,
        "antes":       0,
        "raise_sizes": ["inf",0],
        "num_raises":  [1,0],
        "num_suits":   4,
        "num_ranks":   13,
        "num_hole_cards": 2,
        "num_community_cards": [0,5],
        "num_cards_for_hand": 5,
        "mandatory_num_hole_cards": 0,
        "start_stack": START_STACK,
}

#fix mismatches in different versions of gym
envs.ClubsEnv.metadata["render_modes"] = envs.ClubsEnv.metadata.pop("render.modes", [])

#register the environment with the given configuration.
#also call the gym's register function
envs.register({GAME_ID: CONFIG})
