"""
Generate range cards (13x13 heatmaps) for a trained Poker_DQN model.
One card per position, showing P(all-in) for every starting hand.
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from poker_dqn import ICMNet
from dqn_agent import DuelingDQN

# ── Constants (replicate simconfig / env without heavy imports) ────────────
NUM_PLAYERS = 4
PRIZE_POOL = np.array([1.5, 0.5, -0.5, -1.5], dtype=np.float32)
MAX_STACK_BB = 101  # NUM_PLAYERS * START_STACK // bb + 1 = 4*50//2 + 1
N_RANKS = 13
RANK_LABELS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']


# ── Model loading ───────────────────────────────────────────────────────────

def load_models(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    icm_net = ICMNet(input_dim=8, hidden_dim=128, output_dim=4).to(device)
    icm_net.load_state_dict(checkpoint['icm_net'])
    icm_net.eval()

    dqn_net = DuelingDQN(state_dim=12, hidden_dim=128, n_actions=2).to(device)
    dqn_net.load_state_dict(checkpoint['dqn_net'])
    dqn_net.eval()

    return icm_net, dqn_net


# ── State construction ─────────────────────────────────────────────────────

def build_icm_state(position_idx, stack_bb, other_stacks_bb):
    """
    Build the 8-dim ICM input: 4 normalized stacks + 4-dim raw prize pool.
    Acting player's stack placed at `position_idx`; other_stacks_bb (len 3)
    fills remaining slots in order.
    """
    stacks = [0.0] * NUM_PLAYERS
    stacks[position_idx] = float(stack_bb)
    other = list(other_stacks_bb)
    j = 0
    for i in range(NUM_PLAYERS):
        if i == position_idx:
            continue
        stacks[i] = float(other[j])
        j += 1

    stacks_arr = np.array(stacks, dtype=np.float32)
    stacks_capped = np.minimum(stacks_arr, MAX_STACK_BB - 1)
    stacks_norm = stacks_capped / (MAX_STACK_BB - 1)
    state = np.concatenate([stacks_norm, PRIZE_POOL]).astype(np.float32)
    return state


def build_dqn_state(low_rank, high_rank, suited,
                    stack_bb, active, shortest, position_idx, call_pot):
    """
    Build the 8-dim DQN base state replicating dqn_agent._preprocess_state.
    """
    low_norm = low_rank / (N_RANKS - 1)
    high_norm = high_rank / (N_RANKS - 1)
    suited_norm = float(suited)
    stack_capped = min(stack_bb, MAX_STACK_BB - 1)
    stack_norm = stack_capped / (MAX_STACK_BB - 1)
    active_norm = active / NUM_PLAYERS
    shortest_norm = float(shortest)
    position_norm = position_idx / (NUM_PLAYERS - 1)
    call_norm = np.clip(call_pot, 0.0, 5.0) / 5.0

    return np.array([
        low_norm, high_norm, suited_norm, stack_norm,
        active_norm, shortest_norm, position_norm, call_norm,
    ], dtype=np.float32)


# ── Forward pass ────────────────────────────────────────────────────────────

def compute_allin_prob(dqn_net, dqn_state, icm_out,
                       temperature=1.0, greedy=False):
    """
    Concatenate DQN base state with pre-computed ICM output, run DQN,
    return P(all-in).
    """
    with torch.no_grad():
        dqn_t = torch.from_numpy(dqn_state).unsqueeze(0)
        full = torch.cat([dqn_t, icm_out.unsqueeze(0)], dim=1)  # [1, 12]
        q = dqn_net(full).squeeze(0)  # [2]: Q(all-in), Q(fold)

        if greedy:
            return 1.0 if q[0] > q[1] else 0.0
        probs = torch.softmax(q / temperature, dim=0)
        return probs[0].item()


# ── Range grid ──────────────────────────────────────────────────────────────

def compute_range_grid(icm_net, dqn_net, position_idx,
                       stack_bb, active, shortest, call_pot,
                       other_stacks_bb, temperature, greedy):
    """Build a 13x13 grid of P(all-in) for the given position."""
    grid = np.zeros((13, 13), dtype=np.float32)

    # ICM output is identical across all 169 hands for this position
    icm_state = build_icm_state(position_idx, stack_bb, other_stacks_bb)
    with torch.no_grad():
        icm_out = icm_net(torch.from_numpy(icm_state)).detach()  # [4]

    for row in range(13):
        for col in range(13):
            rank_row = 12 - row  # row 0 → A (rank 12)
            rank_col = 12 - col

            if row == col:
                low_rank = high_rank = rank_row
                suited = 0
            elif row < col:
                # Upper triangle: suited
                high_rank = rank_row
                low_rank = rank_col
                suited = 1
            else:
                # Lower triangle: offsuit
                high_rank = rank_col
                low_rank = rank_row
                suited = 0

            dqn_state = build_dqn_state(
                low_rank, high_rank, suited,
                stack_bb, active, shortest, position_idx, call_pot,
            )
            grid[row, col] = compute_allin_prob(
                dqn_net, dqn_state, icm_out, temperature, greedy,
            )

    return grid


# ── Plotting ────────────────────────────────────────────────────────────────

def hand_label(row, col):
    ranks = 'AKQJT98765432'
    if row == col:
        return ranks[row] + ranks[col]
    if row < col:
        return ranks[row] + ranks[col] + 's'
    return ranks[col] + ranks[row] + 'o'


def plot_range_cards(grids, position_labels, title, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    im = None
    for idx, (ax, grid) in enumerate(zip(axes.flat, grids)):
        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect='equal')
        ax.set_title(position_labels[idx], fontsize=14, fontweight='bold')

        for row in range(13):
            for col in range(13):
                value = grid[row, col]
                label = hand_label(row, col)
                # white text on dark (high or low), black text on mid
                color = 'black' if 0.3 < value < 0.7 else 'white'
                ax.text(col, row, f'{label}\n{value:.0%}',
                        ha='center', va='center', fontsize=6, color=color)

        ax.set_xticks(range(13))
        ax.set_yticks(range(13))
        ax.set_xticklabels(RANK_LABELS)
        ax.set_yticklabels(RANK_LABELS)
        ax.set_xlabel('Second card')
        ax.set_ylabel('First card')

    fig.colorbar(im, ax=axes, shrink=0.6, label='P(All-in)')
    fig.suptitle(f'Range Cards — {title}', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved range cards to {output_path}')


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate range cards from a trained Poker_DQN checkpoint.'
    )
    p.add_argument('--checkpoint', type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints/poker_dqn_2_20260107_173558_609746/final.pt'),
                   help='Path to checkpoint .pt file')
    p.add_argument('--stack-bb', type=float, default=25.0,
                   help='Acting player stack in BB (default: 25)')
    p.add_argument('--active', type=int, default=4,
                   help='Number of active players (default: 4)')
    p.add_argument('--shortest', type=int, default=1, choices=[0, 1],
                   help='Is shortest stack flag (default: 1)')
    p.add_argument('--call-pot', type=float, default=1.0,
                   help='Call-to-pot ratio (default: 1.0)')
    p.add_argument('--temperature', type=float, default=1.0,
                   help='Softmax temperature over Q-values (default: 1.0)')
    p.add_argument('--greedy', action='store_true',
                   help='Use greedy binary decision instead of softmax')
    p.add_argument('--output', type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'results', 'range_cards.png'),
                   help='Output PNG path (default: range_cards.png)')
    p.add_argument('--other-stacks', type=float, nargs=3, default=None,
                   help='Other 3 players stacks in BB (default: equal to --stack-bb)')
    return p.parse_args()


def main():
    args = parse_args()

    icm_net, dqn_net = load_models(args.checkpoint)

    other_stacks = (args.other_stacks
                    if args.other_stacks is not None
                    else [args.stack_bb] * 3)

    grids = []
    position_labels = []
    for pos_idx in range(NUM_PLAYERS):
        grid = compute_range_grid(
            icm_net, dqn_net, pos_idx,
            args.stack_bb, args.active, args.shortest, args.call_pot,
            other_stacks, args.temperature, args.greedy,
        )
        grids.append(grid)
        pos_norm = pos_idx / (NUM_PLAYERS - 1)
        frac = grid.mean()
        position_labels.append(
            f'Position {pos_idx} (norm={pos_norm:.2f})  —  push freq {frac:.1%}'
        )

    mode = 'greedy' if args.greedy else f'softmax T={args.temperature}'
    title = (f'Stack={args.stack_bb}BB, Active={args.active}, '
             f'Shortest={bool(args.shortest)}, Call/Pot={args.call_pot}, '
             f'Mode={mode}')
    plot_range_cards(grids, position_labels, title, args.output)


if __name__ == '__main__':
    main()
