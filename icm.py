import numpy as np
import torch
from poker_dqn import ICMNet, H_PARAMS_ICM

# Constants from simconfig (imported directly to avoid heavy gym/clubs deps)
PRIZE_POOL = (1.5, 0.5, -0.5, -1.5)
NUM_PLAYERS = 4
START_STACK = 50


# ── Exact ICM (Malmuth-Harville) ────────────────────────────────────────────

def compute_icm(stacks, prizes):
    """
    Compute exact ICM equity for each player using the Malmuth-Harville
    recursive method.

    Args:
        stacks: array-like of chip counts per player (must be > 0).
        prizes: array-like of prize values per finishing position
                (index 0 = 1st place, index 1 = 2nd place, ...).

    Returns:
        np.ndarray of shape (n_players,) with each player's ICM equity.
    """
    stacks = np.asarray(stacks, dtype=np.float64)
    prizes = np.asarray(prizes, dtype=np.float64)
    n = len(stacks)
    equity = np.zeros(n, dtype=np.float64)

    def _recurse(remaining, position, prob):
        if len(remaining) == 1:
            equity[remaining[0]] += prob * prizes[position]
            return
        total = sum(stacks[i] for i in remaining)
        for i, player in enumerate(remaining):
            p_win = stacks[player] / total
            equity[player] += prob * p_win * prizes[position]
            next_remaining = remaining[:i] + remaining[i + 1:]
            _recurse(next_remaining, position + 1, prob * p_win)

    _recurse(list(range(n)), 0, 1.0)
    return equity


# ── Normalization bridge ────────────────────────────────────────────────────

def normalize_to_simplex(icm_equity, eps=1e-8):
    """Shift ICM equities to be non-negative and normalize to sum to 1."""
    shifted = icm_equity - icm_equity.min() + eps
    return shifted / shifted.sum()


# ── Input preparation (replicates poker_dqn._state_icm) ────────────────────

def prepare_icm_input(stacks_chips, prize_pool, bb=2, max_stack_bb=101):
    """
    Build the 8-dim input tensor for ICMNet, exactly matching
    Poker_DQN._state_icm (poker_dqn.py:114-125).
    """
    stacks_bb = np.array(
        [min(s // bb, max_stack_bb - 1) for s in stacks_chips],
        dtype=np.float32,
    )
    stacks_norm = stacks_bb / (max_stack_bb - 1)
    state = np.concatenate([stacks_norm, prize_pool]).astype(np.float32)
    return torch.from_numpy(state)


# ── Sample generation ───────────────────────────────────────────────────────

def generate_random_stacks(n_samples=50, total=200, n_players=4, seed=42):
    """
    Generate random chip distributions summing to `total`.
    Each player gets at least 2 chips (1 BB).  Seed is fixed for
    reproducibility.
    """
    rng = np.random.default_rng(seed=seed)
    min_chips = 2
    samples = []
    while len(samples) < n_samples:
        fracs = rng.dirichlet([1.0] * n_players)
        raw = fracs * (total - n_players * min_chips) + min_chips
        stacks = np.floor(raw).astype(int)
        remainder = total - stacks.sum()
        indices = rng.choice(n_players, size=int(remainder), replace=True)
        for idx in indices:
            stacks[idx] += 1
        if all(s >= min_chips for s in stacks):
            samples.append(stacks.astype(np.float64))
    return samples


# ── Spearman rank correlation (no scipy) ────────────────────────────────────

def _rankdata(x):
    """Simple ranking (no ties handling needed for our use case)."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def spearman_rho(a, b):
    ra, rb = _rankdata(a), _rankdata(b)
    n = len(a)
    d2 = ((ra - rb) ** 2).sum()
    return 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))


# ── Comparison logic ────────────────────────────────────────────────────────

def run_comparison(checkpoint_path, n_random=50):
    prize_pool = np.array(PRIZE_POOL, dtype=np.float32)
    total_chips = NUM_PLAYERS * START_STACK

    # Load learned model
    icm_net = ICMNet(
        H_PARAMS_ICM['N_INPUTS'],
        H_PARAMS_ICM['HIDDEN_DIM'],
        H_PARAMS_ICM['N_OUTPUTS'],
    )
    icm_net.load(checkpoint_path, map_location="cpu")
    icm_net.eval()

    # Generate samples: random + edge cases
    samples = generate_random_stacks(n_random, total=total_chips)
    edge_cases = [
        np.array([50.0, 50.0, 50.0, 50.0]),   # equal
        np.array([194.0, 2.0, 2.0, 2.0]),      # one dominant
        np.array([98.0, 98.0, 2.0, 2.0]),       # two dominant
        np.array([80.0, 60.0, 40.0, 20.0]),     # descending
    ]
    samples.extend(edge_cases)

    # Collect results
    maes = []
    rhos = []
    rank_matches = 0
    rows = []

    for stacks in samples:
        exact_raw = compute_icm(stacks, prize_pool.astype(np.float64))
        exact_norm = normalize_to_simplex(exact_raw)

        inp = prepare_icm_input(stacks, prize_pool)
        with torch.no_grad():
            learned = icm_net(inp).numpy()

        mae = np.mean(np.abs(learned - exact_norm))
        rho = spearman_rho(learned, exact_norm)
        rank_ok = np.array_equal(np.argsort(learned), np.argsort(exact_norm))

        maes.append(mae)
        rhos.append(rho)
        rank_matches += int(rank_ok)
        rows.append((stacks, exact_raw, exact_norm, learned, mae, rank_ok))

    n_total = len(samples)
    maes = np.array(maes)
    rhos = np.array(rhos)

    # ── Print results ────────────────────────────────────────────────────

    print("=" * 78)
    print("  ICM Comparison: Learned ICMNet vs Exact Malmuth-Harville")
    print("=" * 78)
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Samples    : {n_total} ({n_random} random + {len(edge_cases)} edge cases)")
    print(f"  Prize pool : {list(PRIZE_POOL)}")
    print()

    print("--- Aggregate Metrics ---")
    print(f"  Mean MAE          : {maes.mean():.4f} +/- {maes.std():.4f}")
    print(f"  Rank match        : {rank_matches} / {n_total}"
          f"  ({100 * rank_matches / n_total:.1f}%)")
    print(f"  Mean Spearman rho : {rhos.mean():.4f} +/- {rhos.std():.4f}")
    print()

    # Per-sample table
    hdr = (f"{'#':>3} | {'Stacks':<22} | {'Exact ICM (raw)':<26} "
           f"| {'Exact (norm)':<26} | {'Learned (softmax)':<26} "
           f"| {'MAE':>6} | Rank")
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    def _fmt(arr):
        return "[" + " ".join(f"{v:6.3f}" for v in arr) + "]"

    for i, (stacks, raw, norm, learned, mae, rok) in enumerate(rows, 1):
        stk = "[" + " ".join(f"{int(s):3d}" for s in stacks) + "]"
        tag = "OK" if rok else "X"
        print(f"{i:3d} | {stk:<22} | {_fmt(raw):<26} "
              f"| {_fmt(norm):<26} | {_fmt(learned):<26} "
              f"| {mae:6.4f} | {tag}")

    print(sep)


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_comparison(
        "checkpoints/poker_dqn_2_20260107_173558_609746/final.pt"
    )
