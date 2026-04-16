from tqdm import tqdm
import numpy as np
from clubs import poker
from qagent import QAgent
from dqn_agent import DQNAgent
from poker_dqn import Poker_DQN


def run_tournament(env, agents, evaluate=False, collect_transitions=False):

    transitions = []

    obs, _ = env.reset(options={"reset_button": True, "reset_stacks": True})
    pre_hand_stacks = list(obs['stacks'])
    done = False

    # Per-player pending transitions for this hand
    pending = {}  # player_idx -> (state, action)

    while not done:

        player_idx = env.table.dealer.action
        curr_agent = agents[player_idx]

        # action selection
        if isinstance(curr_agent, (QAgent, DQNAgent, Poker_DQN)):
            state = curr_agent._preprocess_state(obs)
            action = curr_agent.act(state)
        else:
            action = curr_agent.act(obs)

        # store pending transition for this player
        if isinstance(curr_agent, (DQNAgent, Poker_DQN)):
            pending[player_idx] = (state, action)

        next_obs, reward, done, _, info = env.step(action)
        hand_done = info.get('hand_done', False)

        # finalize transitions when a hand ends
        if hand_done or done:
            bb = env.table.dealer.blinds[1]
            cap = env.max_stack_bb - 1

            if done:
                # tournament over: use prize pool rewards
                hand_rewards = reward
            else:
                # hand over: net chip change over the hand
                post_payout = info['post_payout_stacks']
                hand_rewards = [
                    int(np.clip((post_payout[i] - pre_hand_stacks[i]) // bb, -cap, cap))
                    for i in range(env.num_players)
                ]

            # finalize all pending transitions with done=True
            for pidx, (pstate, paction) in pending.items():
                pagent = agents[pidx]
                if isinstance(pagent, (DQNAgent, Poker_DQN)):
                    if collect_transitions:
                        transitions.append((pstate, paction, hand_rewards[pidx], pstate, True))
                    elif not evaluate:
                        pagent.update_parameters(
                            pstate, paction, hand_rewards[pidx], pstate, True
                        )

            pending.clear()

            if not done:
                # new hand: update baseline stacks (post-blind from fresh obs)
                pre_hand_stacks = list(next_obs['stacks'])

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