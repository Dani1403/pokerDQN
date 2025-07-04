import gymnasium as gym
import sys 
sys.modules["gym"] = gym #fix gymnasium-gym mismatches in registers
from clubs_gym import envs

GAME_ID = "Poker-v0"                      

#configuration of the simulation
CONFIG = {
        "num_players": 6,
        "num_streets": 2,
        "blinds":      [1, 2, 0, 0, 0, 0],
        "antes":       0,
        "raise_sizes": ["inf",0],
        "num_raises":  [1,0],
        "num_suits":   4,
        "num_ranks":   13,
        "num_hole_cards": 2,
        "num_community_cards": [0,5],
        "num_cards_for_hand": 5,
        "mandatory_num_hole_cards": 0,
        "start_stack": 150,
}
#fix mismatches in different versions of gym
envs.ClubsEnv.metadata["render_modes"] = envs.ClubsEnv.metadata.pop("render.modes", [])

#register the environment with the given configuration.
#also call the gym's register function
envs.register({GAME_ID: CONFIG})
