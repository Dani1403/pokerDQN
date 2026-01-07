from tqdm import tqdm
import numpy as np
from clubs import poker
from qagent import QAgent
from dqn_agent import DQNAgent
from poker_dqn import Poker_DQN


def run_tournament(env, agents, evaluate=False, collect_transitions=False):

    transitions = []

    obs, _ = env.reset(options={"reset_button": True, "reset_stacks": True})
    prev_stacks = obs['stacks'].copy()
    done = False

    while not done:

        player_idx = env.table.dealer.action
        curr_agent = agents[player_idx]

        #acton selection
        if isinstance(curr_agent, (QAgent, DQNAgent, Poker_DQN)):
            state = curr_agent._preprocess_state(obs)
            action = curr_agent.act(state)
        else:
            action = curr_agent.act(obs)

        next_obs, reward, done, _, _ = env.step(action)

        # Only compute next state for SAME PLAYER
        next_state = None
        if next_obs is not None and isinstance(curr_agent, (QAgent, DQNAgent, Poker_DQN)):
            next_state = curr_agent._preprocess_state(next_obs)

        # reward shaping
        new_stacks = next_obs['stacks'].copy()
        stack_diff = [new_stacks[i] - prev_stacks[i] for i in range(env.num_players)]
        if not done:
            reward = stack_diff
            bb = env.table.dealer.blinds[1]
            cap = env.max_stack_bb - 1
            reward = [int(np.clip(r // bb, -cap, cap)) for r in stack_diff]

        # Training step or collect transition
        if isinstance(curr_agent, (DQNAgent, Poker_DQN)):
            if collect_transitions:
                transitions.append((state, action, reward[player_idx], next_state, done))

            elif not evaluate:
                curr_agent.update_parameters(
                    state,
                    action,
                    reward[player_idx],
                    next_state,
                    done
                )

        prev_stacks = new_stacks
        obs = next_obs
    if collect_transitions:
        return transitions
    else:
        return reward

""" must provide a fixed lineup """
def run_n_tournaments(env, n_tournaments, evaluate=False, fixed_lineup=None, desc=None, show_tqdm=True):

    rewards_per_tournament = []
    saved_epsilons = {}

    if evaluate:
        for agent in fixed_lineup:
            if hasattr(agent, "epsilon"):
                saved_epsilons[agent] = agent.epsilon
                agent.epsilon = 0.01

    iterator = range(n_tournaments)
    if show_tqdm:
        iterator = tqdm(iterator, ascii=True, ncols=80, desc=desc)

    for _ in iterator:

        reward = run_tournament(env, fixed_lineup, evaluate)

        if evaluate:
            rewards_per_tournament.append(reward)

    if evaluate:
        for agent, eps in saved_epsilons.items():
            agent.epsilon = eps

    return rewards_per_tournament
