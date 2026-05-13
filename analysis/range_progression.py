"""
Track how each agent's push/fold range evolves across training checkpoints.

Outputs:
  1. progression_summary.png  — push frequency vs training step (line plot)
  2. progression_agent{N}.png — milestone range card grids (heatmaps)
"""
import argparse
import glob
import os
import re
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

# ── Constants ───────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RUN_DIR_PATTERN = os.path.join(PROJECT_ROOT, "checkpoints/poker_dqn_{agent}_20260416_155300_916856")
AGENTS = [1, 2, 3, 4]

MILESTONES = [
    100_000, 500_000, 1_000_000, 1_500_000, 2_000_000,
    2_500_000, 3_000_000, 3_500_000, 4_000_000, 4_500_000, 5_000_000,
]

POSITION_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
POSITION_LABELS = [
    f'Position {i} (norm={i / (NUM_PLAYERS - 1):.2f})'
    for i in range(NUM_PLAYERS)
]


# ── Checkpoint discovery ────────────────────────────────────────────────────

def find_checkpoints(agent_id, max_tournaments=5_000_000):
    """
    Return sorted list of (tournament_count, path) for an agent's
    checkpoints up to max_tournaments.
    """
    run_dir = RUN_DIR_PATTERN.format(agent=agent_id)
    pattern = os.path.join(run_dir, "iter_*.pt")
    results = []
    for path in glob.glob(pattern):
        m = re.search(r'iter_\d+_(\d+)\.pt$', path)
        if m:
            t = int(m.group(1))
            if t <= max_tournaments:
                results.append((t, path))
    results.sort()
    return results


# ── Bulk computation ────────────────────────────────────────────────────────

def compute_progression(agent_id, max_tournaments, stack_bb, active,
                        shortest, call_pot, other_stacks, temperature,
                        greedy):
    """
    Load every checkpoint for an agent, compute range grids, return:
      push_freqs  — dict  {tournament: [freq_pos0, ..., freq_pos3]}
      milestone_grids — dict {tournament: [grid_pos0, ..., grid_pos3]}
    """
    checkpoints = find_checkpoints(agent_id, max_tournaments)
    milestone_set = set(MILESTONES)

    push_freqs = {}
    milestone_grids = {}

    for idx, (t, path) in enumerate(checkpoints):
        print(f"  Agent {agent_id}: checkpoint {idx + 1}/{len(checkpoints)}"
              f"  ({t:,} tournaments)")
        icm_net, dqn_net = load_models(path)
        freqs = []
        grids = []
        for pos in range(NUM_PLAYERS):
            grid = compute_range_grid(
                icm_net, dqn_net, pos,
                stack_bb, active, shortest, call_pot,
                other_stacks, temperature, greedy,
            )
            freqs.append(float(grid.mean()))
            if t in milestone_set:
                grids.append(grid)
        push_freqs[t] = freqs
        if grids:
            milestone_grids[t] = grids

    return push_freqs, milestone_grids


# ── Summary line plot ───────────────────────────────────────────────────────

def plot_summary(all_push_freqs, title_suffix, output_path):
    """
    2x2 subplot — one per agent.
    Each subplot has 4 lines (one per position) showing push freq vs step.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)

    for agent_idx, agent_id in enumerate(AGENTS):
        ax = axes.flat[agent_idx]
        data = all_push_freqs[agent_id]
        steps = sorted(data.keys())
        x = [s / 1_000_000 for s in steps]  # display in millions

        for pos in range(NUM_PLAYERS):
            y = [data[s][pos] for s in steps]
            ax.plot(x, y, color=POSITION_COLORS[pos],
                    label=POSITION_LABELS[pos], linewidth=1.5, marker='.',
                    markersize=3)

        ax.set_title(f'Agent {agent_id}', fontsize=13, fontweight='bold')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Tournaments (millions)')
        ax.set_ylabel('Push frequency')

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f'Push Frequency Progression — {title_suffix}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved summary to {output_path}')


# ── Progression grid (one agent) ───────────────────────────────────────────

def _fmt_tournaments(t):
    if t >= 1_000_000:
        return f'{t / 1_000_000:.1f}M'
    return f'{t // 1_000:,}k'


def plot_progression_grid(agent_id, milestone_grids, output_path):
    """
    Large figure: rows = milestones, columns = 4 positions.
    Each cell is a 13x13 range heatmap.
    """
    available = sorted(t for t in MILESTONES if t in milestone_grids)
    n_rows = len(available)
    if n_rows == 0:
        print(f'  Agent {agent_id}: no milestone checkpoints found, skipping.')
        return

    n_cols = NUM_PLAYERS
    cell_size = 4.5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_size * n_cols + 2, cell_size * n_rows + 2),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    im = None
    for ri, t in enumerate(available):
        grids = milestone_grids[t]
        for ci in range(n_cols):
            ax = axes[ri, ci]
            grid = grids[ci]
            im = ax.imshow(grid, cmap=cmap, norm=norm, aspect='equal')

            # annotate cells
            for row in range(13):
                for col in range(13):
                    v = grid[row, col]
                    label = hand_label(row, col)
                    color = 'black' if 0.3 < v < 0.7 else 'white'
                    ax.text(col, row, f'{label}\n{v:.0%}',
                            ha='center', va='center', fontsize=4, color=color)

            ax.set_xticks(range(13))
            ax.set_yticks(range(13))
            ax.set_xticklabels(RANK_LABELS, fontsize=5)
            ax.set_yticklabels(RANK_LABELS, fontsize=5)

            if ri == 0:
                freq = float(grid.mean())
                ax.set_title(POSITION_LABELS[ci], fontsize=9, fontweight='bold')
            if ci == 0:
                freq = float(grid.mean())
                ax.set_ylabel(f'{_fmt_tournaments(t)}\npush {freq:.0%}',
                              fontsize=8, fontweight='bold')

    fig.colorbar(im, ax=axes, shrink=0.4, label='P(All-in)', pad=0.02)
    fig.suptitle(f'Range Progression — Agent {agent_id}',
                 fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved progression grid to {output_path}')


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Track range card evolution across training checkpoints.')
    p.add_argument('--max-tournaments', type=int, default=5_000_000)
    p.add_argument('--stack-bb', type=float, default=25.0)
    p.add_argument('--active', type=int, default=4)
    p.add_argument('--shortest', type=int, default=1, choices=[0, 1])
    p.add_argument('--call-pot', type=float, default=1.0)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--greedy', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    other_stacks = [args.stack_bb] * 3

    mode = 'greedy' if args.greedy else f'softmax T={args.temperature}'
    title_suffix = (f'Stack={args.stack_bb}BB, Active={args.active}, '
                    f'Shortest={bool(args.shortest)}, Mode={mode}')

    all_push_freqs = {}
    all_milestone_grids = {}

    for agent_id in AGENTS:
        print(f'\n=== Agent {agent_id} ===')
        push_freqs, milestone_grids = compute_progression(
            agent_id, args.max_tournaments,
            args.stack_bb, args.active, args.shortest, args.call_pot,
            other_stacks, args.temperature, args.greedy,
        )
        all_push_freqs[agent_id] = push_freqs
        all_milestone_grids[agent_id] = milestone_grids

    # Output 1: summary line plot
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    print('\nPlotting summary...')
    plot_summary(all_push_freqs, title_suffix,
                 os.path.join(results_dir, 'progression_summary.png'))

    # Output 2: progression grids per agent
    for agent_id in AGENTS:
        print(f'Plotting progression grid for agent {agent_id}...')
        plot_progression_grid(
            agent_id,
            all_milestone_grids[agent_id],
            os.path.join(results_dir, f'progression_agent{agent_id}.png'),
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
