"""
Generate range card snapshots for each agent at selected training checkpoints.
Each image shows a 2x2 grid of 13x13 heatmaps (P(all-in) averaged across
all positions) at 4 milestones.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from range_cards import (
    load_models, compute_range_grid, hand_label,
    RANK_LABELS, NUM_PLAYERS,
)
from range_progression import find_checkpoints, _fmt_tournaments

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_MILESTONES = [1_000_000, 2_000_000, 3_500_000, 5_000_000]
AGENTS = [1, 2, 3, 4]


# ── Core computation ───────────────────────────────────────────────────────

def compute_avg_grid(checkpoint_path, stack_bb, active, shortest, call_pot,
                     other_stacks, temperature, greedy):
    """Load a checkpoint and return the position-averaged 13x13 range grid."""
    icm_net, dqn_net = load_models(checkpoint_path)
    grids = []
    for pos in range(NUM_PLAYERS):
        grid = compute_range_grid(
            icm_net, dqn_net, pos,
            stack_bb, active, shortest, call_pot,
            other_stacks, temperature, greedy,
        )
        grids.append(grid)
    return np.mean(grids, axis=0)


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_snapshot(agent_id, snapshot_data, title_suffix, output_path):
    """
    2x2 heatmap grid — one subplot per checkpoint milestone.
    snapshot_data: list of (milestone, grid) tuples (up to 4).
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    im = None
    for idx, ax in enumerate(axes.flat):
        if idx >= len(snapshot_data):
            ax.set_visible(False)
            continue

        milestone, grid = snapshot_data[idx]
        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect='equal')

        freq = float(grid.mean())
        ax.set_title(f'{_fmt_tournaments(milestone)} — push {freq:.0%}',
                     fontsize=14, fontweight='bold')

        for row in range(13):
            for col in range(13):
                value = grid[row, col]
                label = hand_label(row, col)
                color = 'black' if 0.3 < value < 0.7 else 'white'
                ax.text(col, row, f'{label}\n{value:.0%}',
                        ha='center', va='center', fontsize=6, color=color)

        ax.set_xticks(range(13))
        ax.set_yticks(range(13))
        ax.set_xticklabels(RANK_LABELS)
        ax.set_yticklabels(RANK_LABELS)
        ax.set_xlabel('Second card')
        ax.set_ylabel('First card')

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label='P(All-in)')
    fig.suptitle(f'Agent {agent_id} — Range Snapshots (position avg)\n{title_suffix}',
                 fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Range card snapshots at selected training checkpoints.')
    p.add_argument('--milestones', type=int, nargs='+',
                   default=DEFAULT_MILESTONES,
                   help='Up to 4 checkpoint step counts (default: 1M 2M 3.5M 5M)')
    p.add_argument('--agents', type=int, nargs='+', default=AGENTS,
                   help='Which agents to process (default: 1 2 3 4)')
    p.add_argument('--stack-bb', type=float, default=25.0)
    p.add_argument('--active', type=int, default=4)
    p.add_argument('--shortest', type=int, default=1, choices=[0, 1])
    p.add_argument('--call-pot', type=float, default=1.0)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--greedy', action='store_true')
    p.add_argument('--output-dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', 'results'))
    return p.parse_args()


def main():
    args = parse_args()
    milestones = args.milestones[:4]  # 2x2 grid supports at most 4
    other_stacks = [args.stack_bb] * 3

    mode = 'greedy' if args.greedy else f'softmax T={args.temperature}'
    title_suffix = (f'Stack={args.stack_bb}BB, Active={args.active}, '
                    f'Shortest={bool(args.shortest)}, Mode={mode}')

    for agent_id in args.agents:
        print(f'\n=== Agent {agent_id} ===')

        checkpoints = find_checkpoints(agent_id, max_tournaments=max(milestones))
        ckpt_lookup = {t: path for t, path in checkpoints}

        snapshot_data = []
        for ms in milestones:
            if ms not in ckpt_lookup:
                print(f'  WARNING: milestone {ms:,} not found, skipping')
                continue
            print(f'  Loading {_fmt_tournaments(ms)} ...')
            grid = compute_avg_grid(
                ckpt_lookup[ms],
                args.stack_bb, args.active, args.shortest, args.call_pot,
                other_stacks, args.temperature, args.greedy,
            )
            snapshot_data.append((ms, grid))

        if not snapshot_data:
            print(f'  No checkpoints found for agent {agent_id}, skipping.')
            continue

        output_path = os.path.join(
            args.output_dir, f'range_snapshot_agent{agent_id}.png')
        plot_snapshot(agent_id, snapshot_data, title_suffix, output_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
