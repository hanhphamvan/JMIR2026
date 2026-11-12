"""
Step 2: Pure From-Scratch Metaheuristic Policy Optimization (Zero Pre-Loaded Weights).
Trains every policy solely on March-April (Wave 1) and June (Wave 2) stratified days,
under the exact equal budget of 112 objective evaluations per seed across 6 seeds.
Uses multi-core ProcessPoolExecutor for fast CPU execution without GPU dependencies.
"""

import concurrent.futures
import json
import os
import time
from typing import Callable, Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize

from des_engine import PARAM_BOUNDS, DayWorkload, SimConfig, simulate_day
import importlib
cal_mod = importlib.import_module("02_calibrate_twin_parameters")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CALIB_JSON = os.path.join(OUT_DIR, "calibrated_params.json")
WEIGHTS_JSON = os.path.join(OUT_DIR, "learned_weights.json")

SEEDS = [101, 202, 303, 404, 505, 606]
LAMBDA_90 = 0.35
LAMBDA_SLA = 90.0


def clip_bounds(w: np.ndarray) -> np.ndarray:
    return np.clip(w, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])


def sample_uniform(rng: np.random.Generator) -> np.ndarray:
    return PARAM_BOUNDS[:, 0] + rng.random(len(PARAM_BOUNDS)) * (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0])


def stratified_days(workloads: Dict[str, DayWorkload], ym_list: List[str], n_pick: int = 4) -> List[DayWorkload]:
    """Deterministically pick workload-stratified days inside given months (no held-out leakage)."""
    cands = [w for _, w in sorted(workloads.items()) if w.year_month in ym_list and w.n_encounters >= 20]
    cands.sort(key=lambda x: x.n_encounters)
    if len(cands) <= n_pick:
        return cands
    idxs = [int(round(q * (len(cands) - 1))) for q in np.linspace(0.20, 0.92, n_pick)]
    return [cands[i] for i in idxs]


def compute_loss(
    w: np.ndarray,
    days: List[DayWorkload],
    cfg: SimConfig,
    policy: str = "MEMETIC",
    lambda_var: float = 0.25,
    use_normalized: bool = True,
    use_synergy: bool = True,
    use_quadratic_aging: bool = True,
    lambda_90: float = 0.35,
    lambda_sla: float = 90.0,
) -> float:
    """Equation 10-11 multi-objective scalarized daily loss across D_opt."""
    total = 0.0
    for d in days:
        out = simulate_day(
            d, cfg, policy=policy, w=w,
            use_normalized=use_normalized,
            use_synergy=use_synergy,
            use_quadratic_aging=use_quadratic_aging,
        )
        total += out.mean_tat + lambda_90 * out.p90_tat + lambda_sla * out.sla_rate + (lambda_var / 120.0) * out.var_tat
    return total / max(1, len(days))


def optimize_ga(
    eval_fn: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    total_budget: int,
    nm_budget: int = 0,
    warm_seed_w: np.ndarray = None,
) -> Tuple[np.ndarray, float, int]:
    """Real-coded GA (pop=8, tournament k=3, BLX-alpha=0.35, Gaussian mutation) + optional Nelder-Mead local search."""
    ga_budget = total_budget - nm_budget
    pop_size = 8
    evals_done = 0
    span = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]

    pop: List[np.ndarray] = []
    if warm_seed_w is not None:
        pop.append(clip_bounds(warm_seed_w.copy()))
        for _ in range(5):
            pop.append(clip_bounds(warm_seed_w + rng.normal(0.0, 0.15 * span)))
        while len(pop) < pop_size:
            pop.append(sample_uniform(rng))
    else:
        for _ in range(pop_size):
            pop.append(sample_uniform(rng))

    fitness = []
    for ind in pop:
        if evals_done >= ga_budget:
            break
        fitness.append(eval_fn(ind))
        evals_done += 1

    best_idx = int(np.argmin(fitness))
    best_w = pop[best_idx].copy()
    best_f = fitness[best_idx]

    while evals_done < ga_budget:
        new_pop = [best_w.copy()]
        new_fit = [best_f]
        while len(new_pop) < pop_size and evals_done < ga_budget:
            def pick_parent():
                cand_idxs = rng.integers(0, len(pop), size=3)
                return pop[min(cand_idxs, key=lambda idx: fitness[idx])]

            p1, p2 = pick_parent(), pick_parent()
            alpha = 0.35
            lo = np.minimum(p1, p2)
            hi = np.maximum(p1, p2)
            diff = hi - lo
            child = (lo - alpha * diff) + rng.random(len(p1)) * ((1.0 + 2.0 * alpha) * diff)
            mut_mask = rng.random(len(p1)) < 0.25
            if np.any(mut_mask):
                child[mut_mask] += rng.normal(0.0, 0.12 * span[mut_mask])
            child = clip_bounds(child)
            f_child = eval_fn(child)
            evals_done += 1
            new_pop.append(child)
            new_fit.append(f_child)
            if f_child < best_f:
                best_f = f_child
                best_w = child.copy()
        pop, fitness = new_pop, new_fit

    if nm_budget > 0:
        nm_calls = 0

        def nm_obj(x):
            nonlocal nm_calls, evals_done, best_f, best_w
            if nm_calls >= nm_budget:
                return best_f + 1e4
            xc = clip_bounds(np.asarray(x, dtype=float))
            nm_calls += 1
            evals_done += 1
            val = eval_fn(xc)
            if val < best_f:
                best_f = val
                best_w = xc.copy()
            return val

        init_simplex = np.vstack([best_w] + [
            clip_bounds(best_w + np.eye(len(best_w))[i] * (0.08 * span[i]))
            for i in range(len(best_w))
        ])
        try:
            minimize(
                nm_obj, best_w, method="Nelder-Mead",
                options={"initial_simplex": init_simplex, "maxfev": nm_budget, "xatol": 1e-3, "fatol": 1e-3}
            )
        except Exception:
            pass

    return best_w, float(best_f), evals_done


def optimize_de(eval_fn, rng, total_budget, warm_seed_w=None):
    pop_size = 8
    F, CR = 0.6, 0.7
    evals_done = 0
    span = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]

    pop = []
    if warm_seed_w is not None:
        pop.append(clip_bounds(warm_seed_w.copy()))
        for _ in range(5):
            pop.append(clip_bounds(warm_seed_w + rng.normal(0.0, 0.15 * span)))
    while len(pop) < pop_size:
        pop.append(sample_uniform(rng))

    fitness = []
    for ind in pop:
        if evals_done >= total_budget:
            break
        fitness.append(eval_fn(ind))
        evals_done += 1

    best_idx = int(np.argmin(fitness))
    best_w = pop[best_idx].copy()
    best_f = fitness[best_idx]

    while evals_done < total_budget:
        for i in range(pop_size):
            if evals_done >= total_budget:
                break
            choices = [idx for idx in range(pop_size) if idx != i]
            a, b, c = rng.choice(choices, size=3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            j_rand = int(rng.integers(0, len(pop[i])))
            cross = np.array([mutant[j] if (rng.random() < CR or j == j_rand) else pop[i][j] for j in range(len(pop[i]))])
            trial = clip_bounds(cross)
            f_trial = eval_fn(trial)
            evals_done += 1
            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_w = trial.copy()
    return best_w, float(best_f), evals_done


def optimize_rs(eval_fn, rng, total_budget, warm_seed_w=None):
    best_w = sample_uniform(rng)
    best_f = eval_fn(best_w)
    evals_done = 1
    if warm_seed_w is not None and total_budget > 1:
        wc = clip_bounds(warm_seed_w.copy())
        fw = eval_fn(wc)
        evals_done += 1
        if fw < best_f:
            best_f, best_w = fw, wc

    while evals_done < total_budget:
        cand = sample_uniform(rng)
        fc = eval_fn(cand)
        evals_done += 1
        if fc < best_f:
            best_f, best_w = fc, cand
    return best_w, float(best_f), evals_done


def _run_single_seed_task(args):
    """Worker entrypoint for one (algo, seed) task."""
    (algo_name, seed, days_mar, days_apr, days_jun, cfg_w1, cfg_w2,
     policy_mode, lambda_90, lambda_sla, lambda_var, unorm, usyn, uage, nm_budget, method) = args
    rng = np.random.default_rng(seed)

    def eval_mar(w):
        return compute_loss(w, days_mar, cfg_w1, policy_mode, lambda_var, unorm, usyn, uage, lambda_90, lambda_sla)

    def eval_apr(w):
        return compute_loss(w, days_apr, cfg_w1, policy_mode, lambda_var, unorm, usyn, uage, lambda_90, lambda_sla)

    def eval_jun(w):
        return compute_loss(w, days_jun, cfg_w2, policy_mode, lambda_var, unorm, usyn, uage, lambda_90, lambda_sla)

    if method == "ga":
        w_mar, _, _ = optimize_ga(eval_mar, rng, total_budget=56, nm_budget=min(4, nm_budget))
        w_w1, f_w1, _ = optimize_ga(eval_apr, rng, total_budget=56, nm_budget=nm_budget, warm_seed_w=w_mar)
        w_w2, f_w2, _ = optimize_ga(eval_jun, rng, total_budget=112, nm_budget=nm_budget, warm_seed_w=w_w1)
    elif method == "de":
        w_mar, _, _ = optimize_de(eval_mar, rng, total_budget=56)
        w_w1, f_w1, _ = optimize_de(eval_apr, rng, total_budget=56, warm_seed_w=w_mar)
        w_w2, f_w2, _ = optimize_de(eval_jun, rng, total_budget=112, warm_seed_w=w_w1)
    else:
        w_mar, _, _ = optimize_rs(eval_mar, rng, total_budget=56)
        w_w1, f_w1, _ = optimize_rs(eval_apr, rng, total_budget=56, warm_seed_w=w_mar)
        w_w2, f_w2, _ = optimize_rs(eval_jun, rng, total_budget=112, warm_seed_w=w_w1)

    return algo_name, seed, w_w1.tolist(), float(f_w1), w_w2.tolist(), float(f_w2)


def optimize_all():
    t0 = time.time()
    with open(CALIB_JSON, "r", encoding="utf-8") as f:
        calib = json.load(f)

    _, inhouse = cal_mod.load_filtered_inhouse_orders()
    train_subset = inhouse[inhouse["YearMonth"].isin(["2026-03", "2026-04", "2026-06"])].copy()
    workloads = cal_mod.build_daily_workloads(train_subset, calib, t4t5_cap=180.0)

    days_mar = stratified_days(workloads, ["2026-03"], n_pick=4)
    days_apr = stratified_days(workloads, ["2026-04"], n_pick=4)
    days_jun = stratified_days(workloads, ["2026-06"], n_pick=4)

    n_workers = min(16, max(1, (os.cpu_count() or 2) - 2))
    print(f"[TRAIN WINDOWS] March=4d | April=4d | June=4d | Parallel Workers={n_workers} (ZERO May/Jul-Oct access)")

    cfg_w1_lwr = SimConfig(
        prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
        val_service_min=calib["val_service_min"], cadence_mult=calib["wave1_cadence_mult"], routing="lwr"
    )
    cfg_w2_lwr = SimConfig(
        prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
        val_service_min=calib["val_service_min"], cadence_mult=calib["wave2_cadence_mult"], routing="lwr"
    )
    cfg_w1_fix = SimConfig(
        prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
        val_service_min=calib["val_service_min"], cadence_mult=calib["wave1_cadence_mult"], routing="fixed"
    )
    cfg_w2_fix = SimConfig(
        prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
        val_service_min=calib["val_service_min"], cadence_mult=calib["wave2_cadence_mult"], routing="fixed"
    )

    specs = [
        ("Memetic_DT_GA",          cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.35, 90.0,  0.25, True,  True,  True,  10, "ga"),
        ("Memetic_DT_GA_NoVar",    cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.35, 90.0,  0.00, True,  True,  True,  10, "ga"),
        ("Differential_Evolution", cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.35, 90.0,  0.25, True,  True,  True,  0,  "de"),
        ("Random_Search",          cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.35, 90.0,  0.25, True,  True,  True,  0,  "rs"),
        ("Standard_Linear_DT_GA",  cfg_w1_lwr, cfg_w2_lwr, "STD_LINEAR", 0.35, 90.0,  0.25, False, False, False, 0,  "ga"),
        ("Ablation_No_NelderMead", cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.35, 90.0,  0.25, True,  True,  True,  0,  "ga"),
        ("Ablation_No_QuadAging",  cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.35, 90.0,  0.25, True,  True,  False, 10, "ga"),
        ("Ablation_No_Synergy",    cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.35, 90.0,  0.25, True,  False, True,  10, "ga"),
        ("Lambda90_0.20",          cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.20, 90.0,  0.25, True,  True,  True,  10, "ga"),
        ("Lambda90_0.50",          cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.50, 90.0,  0.25, True,  True,  True,  10, "ga"),
        ("LambdaSLA_60",           cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.35, 60.0,  0.25, True,  True,  True,  10, "ga"),
        ("LambdaSLA_120",          cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.35, 120.0, 0.25, True,  True,  True,  10, "ga"),
        ("LambdaVar_0.50",         cfg_w1_lwr, cfg_w2_lwr, "MEMETIC",    0.35, 90.0,  0.50, True,  True,  True,  10, "ga"),
    ]

    tasks = []
    for name, cw1, cw2, pmode, l90, lsla, lvar, unorm, usyn, uage, nmb, mth in specs:
        for seed in SEEDS:
            tasks.append((name, seed, days_mar, days_apr, days_jun, cw1, cw2, pmode, l90, lsla, lvar, unorm, usyn, uage, nmb, mth))

    results_by_algo: Dict[str, List] = {s[0]: [] for s in specs}
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_run_single_seed_task, tasks):
            results_by_algo[res[0]].append(res)

    learned = {}
    for name, *_ in specs:
        rows = sorted(results_by_algo[name], key=lambda x: x[1])  # sort by seed ID
        w1_vecs = [r[2] for r in rows]
        w1_loss = [r[3] for r in rows]
        w2_vecs = [r[4] for r in rows]
        w2_loss = [r[5] for r in rows]
        med1 = int(np.argsort(w1_loss)[len(w1_loss) // 2])
        med2 = int(np.argsort(w2_loss)[len(w2_loss) // 2])
        learned[name] = {
            "algo": name,
            "w1_median_seed_vec": [round(x, 4) for x in w1_vecs[med1]],
            "w2_median_seed_vec": [round(x, 4) for x in w2_vecs[med2]],
            "w1_all_seed_vecs": [[round(x, 4) for x in v] for v in w1_vecs],
            "w2_all_seed_vecs": [[round(x, 4) for x in v] for v in w2_vecs],
            "w1_train_losses": [round(x, 3) for x in w1_loss],
            "w2_train_losses": [round(x, 3) for x in w2_loss],
        }
        print(f"  [OPT DONE] {name:<25} | W1_med_w={learned[name]['w1_median_seed_vec']} | W2_med_w={learned[name]['w2_median_seed_vec']}")

    with open(WEIGHTS_JSON, "w", encoding="utf-8") as f:
        json.dump(learned, f, indent=2)
    print(f"[DONE] Optimized all 8 suites x 6 seeds from scratch in {time.time() - t0:.2f}s -> {WEIGHTS_JSON}")


if __name__ == "__main__":
    optimize_all()
