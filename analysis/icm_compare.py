"""
Compare the learned ICMNet equity function against the exact Malmuth-Harville
ICM formula.  The goal is NOT to check whether ICMNet reproduces exact ICM —
it learns its own representation through self-play.  Instead we compare:

  1. Functional shape — stack sweep curves (monotonicity, diminishing returns)
  2. Ranking agreement — do the two functions agree on who has most/least equity?
  3. Decision impact — same DQN, swap learned ICM for exact ICM: how many
     push/fold decisions change?

Outputs:
  icm_stack_sweep.png   — equity vs stack size for both functions
  icm_scatter.png       — learned vs exact scatter (relationship, not error)
  icm_decision_diff.png — range card difference when swapping ICM source
  Console               — property comparison summary
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from icm import (
    compute_icm, normalize_to_simplex, prepare_icm_input,
    generate_random_stacks, spearman_rho,
)
from range_cards import (
    load_models, build_icm_state, build_dqn_state, compute_allin_prob,
    RANK_LABELS, NUM_PLAYERS, MAX_STACK_BB, PRIZE_POOL as RANGE_PRIZE_POOL,
    hand_label,
)
from range_progression import find_checkpoints, _fmt_tournaments

# ── Constants ──────────────────────────────────────────────────────────────

PRIZE_POOL_RAW = np.array([1.5, 0.5, -0.5, -1.5], dtype=np.float64)
PRIZE_POOL_F32 = PRIZE_POOL_RAW.astype(np.float32)
TOTAL_CHIPS = 200
AGENTS = [1, 2, 3, 4]
AGENT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
BB = 2


# ═══════════════════════════════════════════════════════════════════════════
# 1.  STACK SWEEP — functional shape comparison
# ═══════════════════════════════════════════════════════════════════════════

def stack_sweep(icm_net, player_idx, other_stacks_chips, n_points=60):
    """
    Sweep one player's stack from 2 to 190 chips (1 to 95 BB),
    keeping other stacks fixed.  Total chips are NOT constrained to 200
    since the ICMNet only sees normalized stacks, not absolute totals.

    Returns (stacks_bb, learned_equity, exact_equity_norm, exact_equity_raw).
    """
    chip_range = np.linspace(2, 190, n_points).astype(int)
    chip_range = np.unique(chip_range)

    learned_vals = []
    exact_norm_vals = []
    exact_raw_vals = []
    bb_vals = []

    for chips in chip_range:
        stacks = list(other_stacks_chips)
        stacks.insert(player_idx, float(chips))
        stacks = np.array(stacks, dtype=np.float64)

        # Exact ICM (raw = actual equity, norm = softmax-comparable)
        exact_raw = compute_icm(stacks, PRIZE_POOL_RAW)
        exact_norm = normalize_to_simplex(exact_raw)

        # Learned ICMNet
        inp = prepare_icm_input(stacks, PRIZE_POOL_F32)
        with torch.no_grad():
            learned = icm_net(inp).numpy()

        learned_vals.append(learned[player_idx])
        exact_norm_vals.append(exact_norm[player_idx])
        exact_raw_vals.append(exact_raw[player_idx])
        bb_vals.append(chips // BB)

    return (np.array(bb_vals), np.array(learned_vals),
            np.array(exact_norm_vals), np.array(exact_raw_vals))


def plot_stack_sweeps(agents_data, output_path):
    """
    2x2 grid, one per agent.  Each subplot has two y-axes:
      Left:  Learned ICMNet softmax output (0-1)
      Right: Exact ICM raw equity (actual prize units)
    Both plotted against player 0's stack in BB.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for agent_idx, agent_id in enumerate(AGENTS):
        ax_l = axes.flat[agent_idx]
        bb_vals, learned, exact_norm, exact_raw = agents_data[agent_id]

        # Left axis: learned ICMNet (softmax 0-1)
        l1, = ax_l.plot(bb_vals, learned, color=AGENT_COLORS[agent_idx],
                         linewidth=2, label='Learned ICMNet (softmax)')
        ax_l.set_ylabel('Learned ICMNet output', fontsize=10)
        ax_l.set_ylim(-0.05, 1.05)
        ax_l.tick_params(axis='y', labelcolor=AGENT_COLORS[agent_idx])

        # Right axis: exact ICM raw equity
        ax_r = ax_l.twinx()
        l2, = ax_r.plot(bb_vals, exact_raw, 'k-', linewidth=2,
                         label='Exact ICM (raw equity)')
        l3, = ax_r.plot(bb_vals, exact_norm, 'k--', linewidth=1.5,
                         alpha=0.5, label='Exact ICM (normalized)')
        ax_r.set_ylabel('Exact ICM equity', fontsize=10)

        ax_l.set_xlabel('Player 0 stack (BB)')
        ax_l.set_title(f'Agent {agent_id}', fontsize=13, fontweight='bold')
        ax_l.grid(True, alpha=0.3)

        lines = [l1, l2, l3]
        ax_l.legend(lines, [l.get_label() for l in lines],
                    fontsize=8, loc='upper left')

    fig.suptitle(
        'Stack Sweep — Equity vs Stack Size (player 0)\n'
        'Other 3 players fixed at 50 chips = 25BB',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved stack sweep to {output_path}')


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SCATTER — relationship between learned and exact
# ═══════════════════════════════════════════════════════════════════════════

def compute_scatter_data(icm_net, samples):
    """Return (exact_all, learned_all) arrays for all players across samples."""
    exact_all = []
    learned_all = []
    for stacks in samples:
        exact_raw = compute_icm(stacks, PRIZE_POOL_RAW)
        exact_norm = normalize_to_simplex(exact_raw)
        inp = prepare_icm_input(stacks, PRIZE_POOL_F32)
        with torch.no_grad():
            learned = icm_net(inp).numpy()
        exact_all.extend(exact_norm.tolist())
        learned_all.extend(learned.tolist())
    return np.array(exact_all), np.array(learned_all)


def plot_scatter(agents_scatter, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    for agent_idx, agent_id in enumerate(AGENTS):
        ax = axes.flat[agent_idx]
        exact, learned = agents_scatter[agent_id]

        ax.scatter(exact, learned, alpha=0.3, s=10,
                   color=AGENT_COLORS[agent_idx])
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5,
                label='y = x (identical)')
        ax.set_xlabel('Exact ICM (normalized)')
        ax.set_ylabel('Learned ICMNet')
        ax.set_title(f'Agent {agent_id}', fontsize=13, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        'Learned ICMNet vs Exact ICM — All Players, Random Stacks\n'
        '(each dot = one player in one stack distribution)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved scatter to {output_path}')


# ═══════════════════════════════════════════════════════════════════════════
# 3.  DECISION IMPACT — swap ICM source, compare push/fold range cards
# ═══════════════════════════════════════════════════════════════════════════

def compute_range_grid_with_icm_source(icm_net, dqn_net, position_idx,
                                        stack_bb, active, shortest, call_pot,
                                        other_stacks_bb, temperature, greedy,
                                        use_exact_icm=False):
    """
    Build 13x13 range grid.  If use_exact_icm is True, replace the learned
    ICMNet output with exact ICM (normalized to simplex) before feeding DQN.
    """
    grid = np.zeros((13, 13), dtype=np.float32)

    # Build ICM input
    icm_state = build_icm_state(position_idx, stack_bb, other_stacks_bb)

    if use_exact_icm:
        # Reconstruct chip stacks from BB
        stacks_chips = [0.0] * NUM_PLAYERS
        stacks_chips[position_idx] = float(stack_bb) * BB
        j = 0
        for i in range(NUM_PLAYERS):
            if i == position_idx:
                continue
            stacks_chips[i] = float(other_stacks_bb[j]) * BB
            j += 1
        exact_raw = compute_icm(
            np.array(stacks_chips, dtype=np.float64), PRIZE_POOL_RAW)
        exact_norm = normalize_to_simplex(exact_raw).astype(np.float32)
        icm_out = torch.from_numpy(exact_norm)
    else:
        with torch.no_grad():
            icm_out = icm_net(torch.from_numpy(icm_state)).detach()

    for row in range(13):
        for col in range(13):
            rank_row = 12 - row
            rank_col = 12 - col
            if row == col:
                low_rank = high_rank = rank_row
                suited = 0
            elif row < col:
                high_rank = rank_row
                low_rank = rank_col
                suited = 1
            else:
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


def plot_decision_diff(agents_diff_data, output_path):
    """
    For each agent: show learned range, exact range, and the difference.
    3 columns x 4 rows.
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))

    cmap_range = plt.cm.RdYlGn
    cmap_diff = plt.cm.RdBu_r
    norm_range = mcolors.Normalize(vmin=0.0, vmax=1.0)
    norm_diff = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    for agent_idx, agent_id in enumerate(AGENTS):
        learned_grid, exact_grid, diff_grid = agents_diff_data[agent_id]

        # Column 0: Learned ICM range
        ax0 = axes[agent_idx, 0]
        im0 = ax0.imshow(learned_grid, cmap=cmap_range, norm=norm_range)
        ax0.set_title(f'Agent {agent_id} — Learned ICM', fontsize=11,
                       fontweight='bold')
        _annotate_grid(ax0, learned_grid)

        # Column 1: Exact ICM range
        ax1 = axes[agent_idx, 1]
        im1 = ax1.imshow(exact_grid, cmap=cmap_range, norm=norm_range)
        ax1.set_title(f'Agent {agent_id} — Exact ICM', fontsize=11,
                       fontweight='bold')
        _annotate_grid(ax1, exact_grid)

        # Column 2: Difference (learned - exact)
        ax2 = axes[agent_idx, 2]
        im2 = ax2.imshow(diff_grid, cmap=cmap_diff, norm=norm_diff)
        ax2.set_title(f'Agent {agent_id} — Difference', fontsize=11,
                       fontweight='bold')
        _annotate_grid(ax2, diff_grid, fmt='+.0%', threshold=0.3)

        for ax in [ax0, ax1, ax2]:
            ax.set_xticks(range(13))
            ax.set_yticks(range(13))
            ax.set_xticklabels(RANK_LABELS, fontsize=5)
            ax.set_yticklabels(RANK_LABELS, fontsize=5)

    # Colorbars
    range_axes = [axes[r, c] for r in range(4) for c in range(2)]
    fig.colorbar(im0, ax=range_axes, shrink=0.4,
                 label='P(All-in)', pad=0.02)
    diff_axes = [axes[r, 2] for r in range(4)]
    fig.colorbar(im2, ax=diff_axes, shrink=0.4,
                 label='Learned - Exact', pad=0.02)

    fig.suptitle(
        'Decision Impact: Learned ICM vs Exact ICM Fed to Same DQN\n'
        '(position-averaged, stack=25BB, greedy)',
        fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved decision diff to {output_path}')


def _annotate_grid(ax, grid, fmt='.0%', threshold=0.3):
    for row in range(13):
        for col in range(13):
            v = grid[row, col]
            label = hand_label(row, col)
            color = 'black' if threshold < abs(v) < (1 - threshold) else 'white'
            if fmt == '+.0%':
                txt = f'{label}\n{v:+.0%}'
            else:
                txt = f'{label}\n{v:.0%}'
            ax.text(col, row, txt, ha='center', va='center',
                    fontsize=3.5, color=color)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PROPERTY COMPARISON — console summary
# ═══════════════════════════════════════════════════════════════════════════

def print_property_comparison(agent_id, icm_net, samples):
    """
    Compare key properties of learned ICMNet vs exact ICM:
    - Monotonicity: bigger stack → higher equity?
    - Rank agreement: how often do they rank players the same way?
    - Sensitivity: how much does output change when stacks change?
    """
    rank_agree = 0
    mono_learned = 0
    mono_exact = 0
    n_mono_tests = 0
    rhos = []

    for stacks in samples:
        exact_raw = compute_icm(stacks, PRIZE_POOL_RAW)
        exact_norm = normalize_to_simplex(exact_raw)
        inp = prepare_icm_input(stacks, PRIZE_POOL_F32)
        with torch.no_grad():
            learned = icm_net(inp).numpy()

        # Rank agreement
        if np.array_equal(np.argsort(learned), np.argsort(exact_norm)):
            rank_agree += 1

        # Spearman
        rhos.append(spearman_rho(learned, exact_norm))

        # Monotonicity: for each pair of players, bigger stack → higher equity?
        for i in range(NUM_PLAYERS):
            for j in range(i + 1, NUM_PLAYERS):
                if stacks[i] == stacks[j]:
                    continue
                n_mono_tests += 1
                bigger = i if stacks[i] > stacks[j] else j
                smaller = j if bigger == i else i
                if exact_norm[bigger] >= exact_norm[smaller]:
                    mono_exact += 1
                if learned[bigger] >= learned[smaller]:
                    mono_learned += 1

    n = len(samples)
    print()
    print(f"  === Agent {agent_id} — Property Comparison ===")
    print(f"  Samples: {n}")
    print()
    print(f"  Rank ordering agreement (exact vs learned):  "
          f"{rank_agree}/{n} ({100*rank_agree/n:.1f}%)")
    print(f"  Mean Spearman rho:  {np.mean(rhos):.3f}")
    print()
    print(f"  Monotonicity (bigger stack → higher equity):")
    print(f"    Exact ICM:   {mono_exact}/{n_mono_tests} "
          f"({100*mono_exact/n_mono_tests:.1f}%)")
    print(f"    Learned ICM: {mono_learned}/{n_mono_tests} "
          f"({100*mono_learned/n_mono_tests:.1f}%)")
    print()

    # Show what the learned ICM outputs for a few key scenarios
    key_scenarios = [
        ("Equal stacks", np.array([50.0, 50.0, 50.0, 50.0])),
        ("Descending",   np.array([80.0, 60.0, 40.0, 20.0])),
        ("One dominant",  np.array([140.0, 20.0, 20.0, 20.0])),
        ("One short",     np.array([66.0, 66.0, 66.0, 2.0])),
    ]
    print(f"  Key scenarios:")
    print(f"  {'Scenario':<16} | {'Stacks':<22} | {'Exact (norm)':<28} | {'Learned':<28}")
    print(f"  {'-'*100}")

    def _fmt(arr):
        return "[" + " ".join(f"{v:.3f}" for v in arr) + "]"

    for name, stacks in key_scenarios:
        exact_raw = compute_icm(stacks, PRIZE_POOL_RAW)
        exact_norm = normalize_to_simplex(exact_raw)
        inp = prepare_icm_input(stacks, PRIZE_POOL_F32)
        with torch.no_grad():
            learned = icm_net(inp).numpy()
        stk = "[" + " ".join(f"{int(s):3d}" for s in stacks) + "]"
        print(f"  {name:<16} | {stk:<22} | {_fmt(exact_norm):<28} | {_fmt(learned):<28}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI + Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Compare learned ICMNet vs exact Malmuth-Harville ICM.')
    p.add_argument('--max-tournaments', type=int, default=6_300_000)
    p.add_argument('--n-random', type=int, default=100)
    p.add_argument('--agents', type=int, nargs='+', default=AGENTS)
    p.add_argument('--stack-bb', type=float, default=25.0,
                   help='Stack for decision impact range cards')
    p.add_argument('--output-dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), '..',
                                        'results', 'new_run', 'icm'))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    samples = generate_random_stacks(args.n_random, total=TOTAL_CHIPS)

    print("=" * 70)
    print("  Learned ICMNet vs Exact Malmuth-Harville ICM")
    print("=" * 70)

    # ── 1. Stack sweep ──────────────────────────────────────────────────
    print('\n--- 1. Stack Sweep (equity vs stack size) ---')
    sweep_data = {}
    for agent_id in args.agents:
        checkpoints = find_checkpoints(agent_id, args.max_tournaments)
        if not checkpoints:
            continue
        _, final_path = checkpoints[-1]
        print(f'  Agent {agent_id}: loading final checkpoint...')
        icm_net, _ = load_models(final_path)
        # Sweep player 0's stack from 2..190 chips, others fixed at 50
        sweep_data[agent_id] = stack_sweep(
            icm_net, player_idx=0, other_stacks_chips=[50.0, 50.0, 50.0],
            n_points=80)

    plot_stack_sweeps(sweep_data,
                      os.path.join(args.output_dir, 'icm_stack_sweep.png'))

    # ── 2. Scatter ──────────────────────────────────────────────────────
    print('\n--- 2. Scatter (learned vs exact across random stacks) ---')
    scatter_data = {}
    for agent_id in args.agents:
        checkpoints = find_checkpoints(agent_id, args.max_tournaments)
        if not checkpoints:
            continue
        _, final_path = checkpoints[-1]
        icm_net, _ = load_models(final_path)
        scatter_data[agent_id] = compute_scatter_data(icm_net, samples)

    plot_scatter(scatter_data,
                 os.path.join(args.output_dir, 'icm_scatter.png'))

    # ── 3. Decision impact ──────────────────────────────────────────────
    print('\n--- 3. Decision Impact (range cards: learned vs exact ICM) ---')
    diff_data = {}
    stack_bb = args.stack_bb
    other_stacks_bb = [stack_bb] * 3

    for agent_id in args.agents:
        checkpoints = find_checkpoints(agent_id, args.max_tournaments)
        if not checkpoints:
            continue
        _, final_path = checkpoints[-1]
        print(f'  Agent {agent_id}: computing range grids...')
        icm_net, dqn_net = load_models(final_path)

        # Position-averaged grids
        learned_grids = []
        exact_grids = []
        for pos in range(NUM_PLAYERS):
            lg = compute_range_grid_with_icm_source(
                icm_net, dqn_net, pos, stack_bb, 4, 1, 1.0,
                other_stacks_bb, 1.0, True, use_exact_icm=False)
            eg = compute_range_grid_with_icm_source(
                icm_net, dqn_net, pos, stack_bb, 4, 1, 1.0,
                other_stacks_bb, 1.0, True, use_exact_icm=True)
            learned_grids.append(lg)
            exact_grids.append(eg)

        learned_avg = np.mean(learned_grids, axis=0)
        exact_avg = np.mean(exact_grids, axis=0)
        diff = learned_avg - exact_avg

        n_agree = np.sum((learned_avg > 0.5) == (exact_avg > 0.5))
        print(f'    Push agreement: {n_agree}/169 hands '
              f'({100*n_agree/169:.1f}%)')
        print(f'    Mean |diff|: {np.abs(diff).mean():.3f}')

        diff_data[agent_id] = (learned_avg, exact_avg, diff)

    plot_decision_diff(diff_data,
                       os.path.join(args.output_dir, 'icm_decision_diff.png'))

    # ── 4. Property comparison ──────────────────────────────────────────
    print('\n--- 4. Property Comparison ---')
    for agent_id in args.agents:
        checkpoints = find_checkpoints(agent_id, args.max_tournaments)
        if not checkpoints:
            continue
        _, final_path = checkpoints[-1]
        icm_net, _ = load_models(final_path)
        print_property_comparison(agent_id, icm_net, samples)

    print('\nDone.')


if __name__ == '__main__':
    main()
