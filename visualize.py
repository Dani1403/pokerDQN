import numpy as np
import random
from collections import defaultdict
from clubs.poker.card import Card
from poker_dqn import Poker_DQN
from dqn_agent import DQNAgent
import simulation 

RANKS = "23456789TJQKA"
SUITS = "SHDC"

def generate_all_hands():
    """Return all 169 canonical starting hands (AA, AKs, AKo, etc.)"""
    hands = []
    for i, r1 in enumerate(RANKS[::-1]):
        for j, r2 in enumerate(RANKS[::-1]):
            if j < i:
                continue

            if i == j:
                hands.append((r1, r2, "pair"))
            else:
                hands.append((r1, r2, "offsuit"))
                hands.append((r1, r2, "suited"))
    return hands


def sample_obs(env, hand):
    """Create a realistic observation using your env format"""
    r1, r2, htype = hand

    if htype == "pair":
        c1 = Card(r1 + "S")
        c2 = Card(r2 + "H")
    elif htype == "suited":
        c1 = Card(r1 + "S")
        c2 = Card(r2 + "S")
    else:
        c1 = Card(r1 + "S")
        c2 = Card(r2 + "H")

    # --- simulate stacks (important for your state encoding) ---
    stacks = [
        random.randint(5, env.table.dealer.start_stack)
        for _ in range(env.num_players)
    ]

    action_player = random.randint(0, env.num_players - 1)

    obs = {
        "action": action_player,
        "active": [True] * env.num_players,
        "button": random.randint(0, env.num_players - 1),
        "call": 0,
        "community_cards": [],
        "hole_cards": [c1, c2],
        "max_raise": env.all_in,
        "min_raise": 0,
        "pot": random.randint(0, 20),
        "stacks": stacks,
        "street_commits": [0] * env.num_players,
    }

    return obs


def visualize_policy(agent, env, n_samples=200):
    """
    For each starting hand:
    - sample random states
    - run model
    - compute % all-in
    """

    results = {}

    all_hands = generate_all_hands()

    for hand in all_hands:

        print("Simulation of hand :", hand)

        allin_count = 0

        for _ in range(n_samples):
            obs = sample_obs(env, hand)

            # --- preprocess ---
            state = agent._preprocess_state(obs)

            # --- forward pass ---
            action = agent.act(state)
            if action == env.all_in:
                allin_count += 1

        freq = allin_count / n_samples
        results[hand] = freq

    # --- pretty print ---
    print("\n=== POLICY VISUALIZATION ===\n")
    for (r1, r2, htype), freq in sorted(results.items(), reverse=True):
        label = f"{r1}{r2}"
        if htype == "suited":
            label += "s"
        elif htype == "offsuit":
            label += "o"

        print(f"{label:4s} -> ALL-IN freq: {freq:.2%}")

    return results


def main():
    env = simulation.PokerTournament()
    agent = Poker_DQN(env, 
                      name="poker_dqn_test",
                      enable_tb=False)

    agent = DQNAgent(env,
                     name="dqn_agent_test",
                     enable_tb=False)


    # --- load checkpoint ---
    #checkpoint_dir = "checkpoints/poker_dqn_1_20260107_173558_609746"
    checkpoint_dir = "checkpoints/dqn4_20251222_163157_707965"
    #agent.load(checkpoint_dir + "/final.pt")
    agent.load(checkpoint_dir + "/iter_2_10000.pt")

    visualize_policy(agent, env, n_samples=5_000)


if __name__ == "__main__":
    main()