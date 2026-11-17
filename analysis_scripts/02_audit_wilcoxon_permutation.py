"""
Diagnostic Analysis 2: Sign-Flip Permutation Tests for the Wilcoxon Signed-Rank Statistic,
7-Day Circular Moving-Block Bootstrap (MBB) 95% CIs (Table 6), and Seed-Paired Tests with
2-Level Bootstrap CIs (Table A4).
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats


def sign_flip_permutation_wilcoxon(diffs: np.ndarray, n_permutations: int = 400000, seed: int = 202611) -> float:
    """
    Compute 2-sided sign-flip permutation P value for the Wilcoxon signed-rank statistic,
    excluding zero differences and using mid-ranks (average ranks) for ties.
    Uses exact enumeration of all 2^n sign flips when n <= 20 nonzero days,
    and n_permutations random sign flips otherwise.
    """
    diffs = np.asarray(diffs, dtype=float)
    nonzero = diffs[np.abs(diffs) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    w_obs = float(np.sum(ranks[nonzero > 0]))
    mean_w = 0.5 * float(np.sum(ranks))
    dev_obs = abs(w_obs - mean_w)
    if n <= 20:
        signs = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n)
        w_perm = np.sum(ranks * (signs > 0), axis=1)
        dev_perm = np.abs(w_perm - mean_w)
        return float(np.mean(dev_perm >= dev_obs - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.integers(0, 2, size=(n_permutations, n))
        w_perm = signs @ ranks
        dev_perm = np.abs(w_perm - mean_w)
        return float(np.mean(dev_perm >= dev_obs - 1e-12))


def holm_adjust(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        val = min(1.0, pvals[orig_idx] * (m - rank_idx))
        running_max = max(running_max, val)
        adj[orig_idx] = running_max
    return adj


def two_level_bootstrap_ci(diff_matrix: np.ndarray, block_len: int = 7, n_boot: int = 4000, seed: int = 202611):
    """
    2-level bootstrap over S seeds (resampled with replacement) and D days
    (resampled in circular blocks of length block_len).
    diff_matrix has shape (S, D).
    """
    S, D = diff_matrix.shape
    n_blocks = int(np.ceil(D / block_len))
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        seed_idx = rng.integers(0, S, size=S)
        starts = rng.integers(0, D, size=n_blocks)
        day_idx = np.concatenate([np.arange(s, s + block_len) % D for s in starts])[:D]
        sub = diff_matrix[np.ix_(seed_idx, day_idx)]
        boot_means[b] = float(np.mean(sub))
    return float(np.mean(diff_matrix)), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "outputs", "daily_simulation_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        ds = (df["Memetic_DT_GA_SLA_Pct"] - df["FCFS_SLA_Pct"]).values
        p_raw = sign_flip_permutation_wilcoxon(ds)
        print(f"Memetic DT-GA vs FCFS daily SLA difference: mean={np.mean(ds):.4f} pp, exact sign-flip P={p_raw:.4f}")
    else:
        print("Sign-flip permutation Wilcoxon and 2-level bootstrap module ready.")
