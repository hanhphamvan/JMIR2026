"""
Pure Discrete-Event Simulation (DES) Engine for Outpatient Clinical Laboratory.
Implements the exact Fork-Join queueing physics, 14 analyzer modules across 5 sections,
8 loading staff, 5 validation staff, sequential cadence / coagulation batch admission,
Least-Work-Remaining (LWR) vs Fixed-Index analyzer routing, classical dispatching rules,
Standard Linear DT-GA, and Memetic DT-GA (Equations 1-9).

ZERO hardcoded simulation results or weights live in this module.
"""

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# Section / Resource group IDs
G_BIOCHEM = 0
G_HEMATO = 1
G_COAG = 2
G_IMMUNO = 3
G_URINE = 4

GROUP_NAMES = ["Biochemistry", "Hematology", "Coagulation", "Immunoassay", "Urinalysis"]
GROUP_TO_ID = {name: i for i, name in enumerate(GROUP_NAMES)}

# Event type constants for fast integer comparisons in heapq
EVT_JOB_READY = 0
EVT_TRY_ANALYZER = 1
EVT_JOB_DONE_SEQ = 2
EVT_BATCH_DONE = 3

# Physical analyzer inventory (Table 1 - 14 logical modules across 5 sections)
ANALYZER_SPECS = [
    # Biochemistry (group 0, sequential)
    {"id": 6,  "name": "Advia1650",      "group": G_BIOCHEM, "mode": "seq",   "cap": 84,  "speed": 1.00, "start": "2026-01-01", "end": "2026-10-31"},
    {"id": 18, "name": "AU680",          "group": G_BIOCHEM, "mode": "seq",   "cap": 84,  "speed": 1.00, "start": "2026-01-01", "end": "2026-04-01"},
    {"id": 210,"name": "C501_E601_c501", "group": G_BIOCHEM, "mode": "seq",   "cap": 100, "speed": 1.00, "start": "2026-03-10", "end": "2026-10-31"},
    # Hematology (group 1, sequential; Celltac_6510 retired post-Feb per 0 empirical May-Oct orders)
    {"id": 19, "name": "XN1000",         "group": G_HEMATO,  "mode": "seq",   "cap": 50,  "speed": 1.00, "start": "2026-01-01", "end": "2026-10-31"},
    {"id": 20, "name": "XN550",          "group": G_HEMATO,  "mode": "seq",   "cap": 20,  "speed": 1.00, "start": "2026-01-17", "end": "2026-10-31"},
    {"id": 2,  "name": "Celltac_6510",   "group": G_HEMATO,  "mode": "seq",   "cap": 1,   "speed": 1.00, "start": "2026-01-01", "end": "2026-02-28"},
    # Coagulation (group 2, batch up to 25 tubes per run)
    {"id": 7,  "name": "Coalab",         "group": G_COAG,    "mode": "batch", "cap": 25,  "speed": 1.00, "start": "2026-01-01", "end": "2026-10-31"},
    {"id": 11, "name": "Coalab_1",       "group": G_COAG,    "mode": "batch", "cap": 25,  "speed": 1.00, "start": "2026-01-01", "end": "2026-10-31"},
    {"id": 23, "name": "Coalab_2",       "group": G_COAG,    "mode": "batch", "cap": 25,  "speed": 1.00, "start": "2026-06-13", "end": "2026-10-31"},
    # Immunoassay (group 3, sequential)
    {"id": 15, "name": "E601",           "group": G_IMMUNO,  "mode": "seq",   "cap": 100, "speed": 1.00, "start": "2026-01-01", "end": "2026-10-31"},
    {"id": 17, "name": "E601_1",         "group": G_IMMUNO,  "mode": "seq",   "cap": 100, "speed": 1.00, "start": "2026-01-01", "end": "2026-10-31"},
    {"id": 211,"name": "C501_E601_e601", "group": G_IMMUNO,  "mode": "seq",   "cap": 100, "speed": 1.00, "start": "2026-03-24", "end": "2026-10-31"},
    {"id": 14, "name": "Cobas_D1",       "group": G_IMMUNO,  "mode": "seq",   "cap": 30,  "speed": 1.00, "start": "2026-01-01", "end": "2026-10-31"},
    # Urinalysis (group 4, sequential; Urinalysis ID=3 listed first -> exposes fixed-scan bias)
    {"id": 3,  "name": "Urinalysis",     "group": G_URINE,   "mode": "seq",   "cap": 84,  "speed": 1.00, "start": "2026-01-01", "end": "2026-10-31"},
    {"id": 22, "name": "Uriscan_Super",  "group": G_URINE,   "mode": "seq",   "cap": 50,  "speed": 1.00, "start": "2026-04-06", "end": "2026-10-31"},
]

# Parameter search bounds (Equation 2)
PARAM_BOUNDS = np.array([
    [0.0, 4.0],   # w1: waiting time weight
    [0.0, 4.0],   # w2: processing speed / duration weight
    [0.0, 5.0],   # w3: Fork-Join completion synergy / ratio weight
    [0.0, 6.0],   # w4: remaining sibling / parallel critical path weight
    [-2.0, 2.0],  # w5: SLA slack weight
    [0.0, 5.0],   # w6: quadratic anti-starvation aging / near-breach boost
], dtype=float)


@dataclass
class DayWorkload:
    """Compact container of all retained in-house jobs and encounters for 1 calendar day."""
    order_date: str
    year_month: str
    n_jobs: int
    n_encounters: int
    job_enc: np.ndarray        # int32 [n_jobs] -> encounter idx [0..n_encounters-1]
    job_group: np.ndarray      # int32 [n_jobs] -> resource group [0..4]
    job_t2: np.ndarray         # float64 [n_jobs] -> specimen receipt time (min of day)
    job_pred_dur: np.ndarray   # float64 [n_jobs] -> expected processing duration p_hat (from cal only)
    job_real_dur: np.ndarray   # float64 [n_jobs] -> realized processing duration p_tilde
    enc_first_t2: np.ndarray   # float64 [n_encounters] -> a_i*
    enc_job_counts: np.ndarray # int32 [n_encounters] -> |J_i|
    enc_res_lag: np.ndarray    # float64 [n_encounters] -> empirical residual T4->T5 lag r_i (min)
    obs_tat_cap180: np.ndarray # float64 [valid] -> historical capped TAT for fidelity comparison
    obs_tat_uncap: np.ndarray  # float64 [valid] -> historical uncapped TAT


@dataclass
class SimConfig:
    """Calibrated parameters & scenario configuration."""
    prep_lags: List[float]                # alpha_g for groups 0..4 (minutes)
    cadence_epochs: Dict[str, List[float]]# group_id_str -> [morning, midday, afternoon] cadence (min)
    val_service_min: float = 1.0          # active validation service q (minutes)
    load_service_min: float = 1.0         # loading service q_load (minutes)
    cap_loaders: int = 8                  # loading staff pool
    cap_validators: int = 5               # validation staff pool
    routing: str = "lwr"                  # "lwr" (least-work-remaining) or "fixed" (fixed index scan)
    cadence_mult: float = 1.0             # sensitivity multiplier on cadence
    lag_mult: float = 1.0                 # sensitivity multiplier on prep/residual lag
    shared_c501_e601_gate: bool = False   # stress test: shared 1-min conveyor gate between IDs 210 & 211
    outage_advia1650: bool = False        # stress test: disable Advia1650 (ID=6)
    rerun_prob: float = 0.0               # stress test: analytical rerun rate (e.g. 0.05)
    rerun_mult: float = 2.0               # multiplier on service duration when rerun triggers
    batch_min_wait: float = 5.0           # max batch coalesce window for Coagulation
    force_coalab2_off_may: bool = True    # Coalab_2 starts June 13


@dataclass
class DaySimOutput:
    """Simulation outcomes for one day under one policy."""
    tats: np.ndarray
    mean_tat: float
    median_tat: float
    p90_tat: float
    sla_rate: float
    var_tat: float
    group_mean_t2_t4: List[float] = field(default_factory=list)
    group_t2_t4_sum: List[float] = field(default_factory=list)
    group_t2_t4_cnt: List[int] = field(default_factory=list)
    mean_job_t2_t4: float = 0.0
    mean_job_wait: float = 0.0
    job_t2_t4_sum: float = 0.0
    job_wait_sum: float = 0.0
    job_count_done: int = 0
    analyzer_jobs: Dict[str, int] = field(default_factory=dict)
    analyzer_busy_min: Dict[str, float] = field(default_factory=dict)


def _diurnal_epoch(t_min: float) -> int:
    """Morning surge (<11:00 = 660m), midday (660-840m), afternoon (>=840m)."""
    if t_min < 660.0:
        return 0
    if t_min < 840.0:
        return 1
    return 2


def get_active_analyzers_for_date(date_str: str, outage_advia1650: bool = False) -> List[int]:
    """Return indices into ANALYZER_SPECS active on `date_str`."""
    active = []
    for idx, spec in enumerate(ANALYZER_SPECS):
        if outage_advia1650 and spec["id"] == 6:
            continue
        if spec["start"] <= date_str <= spec["end"]:
            active.append(idx)
    return active


def _score_job(
    policy: str,
    w: Optional[np.ndarray],
    j: int,
    t: float,
    rho_j: float,
    pred_j: float,
    first_a: float,
    tot_enc: int,
    rem_enc: int,
    rem_max_pred: float,
    use_normalized: bool = True,
    use_synergy: bool = True,
    use_quadratic_aging: bool = True,
) -> float:
    """Compute dispatching priority score (HIGHEST score = highest priority)."""
    if policy == "FCFS":
        return -rho_j - j * 1e-6
    if policy == "SPT":
        return -pred_j - rho_j * 1e-5
    if policy == "EDD":
        return -(first_a + 120.0) - rho_j * 1e-6
    if policy == "LST":
        slack = (first_a + 120.0) - t - rem_max_pred
        return -slack - rho_j * 1e-6
    if policy == "HRRN":
        wait = max(0.0, t - rho_j)
        return (wait + pred_j) / max(1.0, pred_j)
    if policy == "COMPLETION":
        done_ratio = (tot_enc - rem_enc) / max(1, tot_enc)
        wait = max(0.0, t - rho_j)
        return 2.5 * done_ratio + wait / 60.0 - pred_j / 60.0

    # Parametric policies (w is a 6D vector)
    wait = max(0.0, t - rho_j)
    elapsed = max(0.0, t - first_a)
    done_ratio = (tot_enc - rem_enc) / max(1, tot_enc)

    if policy == "STD_LINEAR" or not use_normalized:
        # Raw unnormalized scales (minutes vs counts vs indicator)
        f1 = wait
        f2 = -pred_j
        f3 = done_ratio * 10.0
        f4 = float(tot_enc - rem_enc)
        f5 = 120.0 - elapsed
        f6 = 1.0 if elapsed > 90.0 else 0.0
    else:
        # Upgraded Memetic DT-GA normalized [0,1] state features + nonlinear mechanisms
        f1 = min(1.0, wait / 120.0)
        f2 = 1.0 - min(1.0, pred_j / 60.0)
        if use_synergy and tot_enc >= 2 and done_ratio >= 0.749:
            f3 = done_ratio * 2.5  # Nonlinear critical-path boost on final sibling(s)
        else:
            f3 = done_ratio
        f4 = min(1.0, rem_max_pred / 60.0)
        f5 = min(1.0, max(0.0, 120.0 - elapsed) / 120.0)
        if use_quadratic_aging:
            excess = max(0.0, wait - 60.0)
            f6 = min(4.0, (excess / 60.0) ** 2)
        else:
            f6 = 1.0 if wait > 60.0 else 0.0

    return float(
        w[0] * f1 + w[1] * f2 + w[2] * f3 + w[3] * f4 + w[4] * f5 + w[5] * f6 - rho_j * 1e-7
    )


def simulate_day(
    day: DayWorkload,
    cfg: SimConfig,
    policy: str = "FCFS",
    w: Optional[np.ndarray] = None,
    use_normalized: bool = True,
    use_synergy: bool = True,
    use_quadratic_aging: bool = True,
    record_traces: bool = False,
    seed: int = 42,
) -> DaySimOutput:
    """
    Execute a deterministic discrete-event simulation for one complete calendar day.
    Common random numbers ensure all policies face identical arrivals and realized durations.
    """
    n_jobs = day.n_jobs
    n_enc = day.n_encounters
    prep_lags = cfg.prep_lags
    lag_mult = cfg.lag_mult
    cad_mult = cfg.cadence_mult
    cad_epochs = cfg.cadence_epochs

    # Precompute realized durations (with optional stochastic rerun stress)
    real_durs = day.job_real_dur
    if cfg.rerun_prob > 0.0:
        rng = np.random.default_rng(seed)
        rerun_mask = rng.random(n_jobs) < cfg.rerun_prob
        if np.any(rerun_mask):
            real_durs = real_durs.copy()
            real_durs[rerun_mask] *= cfg.rerun_mult

    # Active analyzers grouped by resource section
    active_m_indices = get_active_analyzers_for_date(day.order_date, cfg.outage_advia1650)
    group_analyzers: List[List[int]] = [[] for _ in range(5)]
    for m_idx in active_m_indices:
        group_analyzers[ANALYZER_SPECS[m_idx]["group"]].append(m_idx)

    n_all_m = len(ANALYZER_SPECS)
    onboard_finishes: List[List[float]] = [[] for _ in range(n_all_m)]  # sorted/min-heap of finish times
    next_launch_ready = [0.0] * n_all_m
    batch_busy = [False] * n_all_m
    shared_c501_gate_free = 0.0  # used only if shared_c501_e601_gate=True

    m_jobs_count = [0] * n_all_m
    m_busy_intervals: List[List[Tuple[float, float]]] = [[] for _ in range(n_all_m)] if record_traces else []
    grp_t2_t4_sum = [0.0] * 5
    grp_t2_t4_cnt = [0] * 5
    job_wait_sum = 0.0

    # Loader pool min-heap
    loader_heap = [0.0] * max(1, cfg.cap_loaders)
    heapq.heapify(loader_heap)
    load_dur = cfg.load_service_min

    # Waiting queues per group: list of ready job indices
    waiting_jobs: List[List[int]] = [[] for _ in range(5)]
    pending_wakeup = [1e18] * 5  # deduplicate EVT_TRY_ANALYZER per group to prevent event storms
    enc_rem = day.enc_job_counts.tolist()
    enc_max_pred_rem = [0.0] * n_enc
    # Native Python lists avoid np.float64 boxing overhead inside tight DES hot loops (~10x faster)
    enc_pred_lists: List[List[float]] = [[] for _ in range(n_enc)]
    job_enc = day.job_enc.tolist()
    job_grp = day.job_group.tolist()
    job_t2 = day.job_t2.tolist()
    job_pred = day.job_pred_dur.tolist()
    job_pred_inv_norm = [1.0 - min(1.0, p / 60.0) for p in job_pred]
    real_durs_list = real_durs.tolist()
    enc_first_t2 = day.enc_first_t2.tolist()
    enc_job_counts = day.enc_job_counts.tolist()
    enc_inv_tot = [1.0 / max(1, c) for c in enc_job_counts]
    enc_res_lag = day.enc_res_lag.tolist()

    job_rho = [0.0] * n_jobs
    ev_heap: List[Tuple[float, int, int, int]] = []  # (time, evt_type, arg1, arg2)

    for j in range(n_jobs):
        g = job_grp[j]
        e = job_enc[j]
        p_hat = job_pred[j]
        enc_pred_lists[e].append(p_hat)
        if p_hat > enc_max_pred_rem[e]:
            enc_max_pred_rem[e] = p_hat
        rho = job_t2[j] + prep_lags[g] * lag_mult
        job_rho[j] = rho
        ev_heap.append((rho, EVT_JOB_READY, j, g))
    heapq.heapify(ev_heap)

    # Validation pool (Fork-Join downstream server pool)
    val_heap = [0.0] * max(1, cfg.cap_validators)
    heapq.heapify(val_heap)
    q_val = cfg.val_service_min
    enc_tats = [0.0] * n_enc
    w0 = float(w[0]) if w is not None else 0.0
    w1 = float(w[1]) if w is not None else 0.0
    w2 = float(w[2]) if w is not None else 0.0
    w3 = float(w[3]) if w is not None else 0.0
    w4 = float(w[4]) if w is not None else 0.0
    w5 = float(w[5]) if w is not None else 0.0
    is_parametric = policy in ("MEMETIC", "STD_LINEAR")
    is_linear_raw = (policy == "STD_LINEAR") or (not use_normalized)

    def pick_best_waiting_job(g: int, t: float) -> int:
        q_list = waiting_jobs[g]
        nq = len(q_list)
        if nq == 1:
            return q_list.pop(0)
        if policy == "FCFS":
            best_i = 0
            best_val = job_rho[q_list[0]]
            for k in range(1, nq):
                v = job_rho[q_list[k]]
                if v < best_val:
                    best_val = v
                    best_i = k
            return q_list.pop(best_i)

        if is_parametric and not is_linear_raw:
            # Fast inlined Memetic DT-GA hot scoring loop
            best_i = 0
            best_score = -1e18
            inv120 = 1.0 / 120.0
            inv60 = 1.0 / 60.0
            for k in range(nq):
                j = q_list[k]
                e = job_enc[j]
                rho_j = job_rho[j]
                wait = t - rho_j
                if wait < 0.0:
                    wait = 0.0
                f1 = wait * inv120
                if f1 > 1.0:
                    f1 = 1.0
                f2 = job_pred_inv_norm[j]
                tot_e = enc_job_counts[e]
                rem_e = enc_rem[e]
                done_ratio = (tot_e - rem_e) * enc_inv_tot[e]
                if use_synergy and tot_e >= 2 and done_ratio >= 0.749:
                    f3 = done_ratio * 2.5
                else:
                    f3 = done_ratio
                f4 = enc_max_pred_rem[e] * inv60
                if f4 > 1.0:
                    f4 = 1.0
                elapsed = t - enc_first_t2[e]
                if elapsed < 0.0:
                    elapsed = 0.0
                slack = (120.0 - elapsed) * inv120
                f5 = 0.0 if slack <= 0.0 else (1.0 if slack >= 1.0 else slack)
                if use_quadratic_aging:
                    if wait > 60.0:
                        ex = (wait - 60.0) * inv60
                        f6 = ex * ex
                        if f6 > 4.0:
                            f6 = 4.0
                    else:
                        f6 = 0.0
                else:
                    f6 = 1.0 if wait > 60.0 else 0.0
                sc = w0 * f1 + w1 * f2 + w2 * f3 + w3 * f4 + w4 * f5 + w5 * f6 - rho_j * 1e-7
                if sc > best_score:
                    best_score = sc
                    best_i = k
            return q_list.pop(best_i)

        best_i = 0
        best_score = -1e18
        for k in range(nq):
            j = q_list[k]
            e = job_enc[j]
            sc = _score_job(
                policy, w, j, t, job_rho[j], job_pred[j],
                enc_first_t2[e], enc_job_counts[e], enc_rem[e],
                enc_max_pred_rem[e], use_normalized, use_synergy, use_quadratic_aging
            )
            if sc > best_score:
                best_score = sc
                best_i = k
        return q_list.pop(best_i)

    def select_analyzer_for_group(g: int, t: float) -> Optional[int]:
        """Choose compatible active analyzer ready at time t using LWR or fixed-index scan."""
        cand_list = group_analyzers[g]
        if cfg.routing == "fixed":
            for m_idx in cand_list:
                spec = ANALYZER_SPECS[m_idx]
                if spec["mode"] == "batch":
                    if not batch_busy[m_idx] and next_launch_ready[m_idx] <= t + 1e-9:
                        return m_idx
                else:
                    if len(onboard_finishes[m_idx]) < spec["cap"] and next_launch_ready[m_idx] <= t + 1e-9:
                        return m_idx
            return None

        # Least-Work-Remaining (LWR) routing across compatible available analyzers
        best_m = None
        best_work = 1e18
        for m_idx in cand_list:
            spec = ANALYZER_SPECS[m_idx]
            if spec["mode"] == "batch":
                if not batch_busy[m_idx] and next_launch_ready[m_idx] <= t + 1e-9:
                    # Break ties cleanly by fewest historical admissions + remaining work
                    work = m_jobs_count[m_idx] * 0.001
                    if work < best_work:
                        best_work = work
                        best_m = m_idx
            else:
                onb = onboard_finishes[m_idx]
                if len(onb) < spec["cap"] and next_launch_ready[m_idx] <= t + 1e-9:
                    # Sum of residual onboard processing time normalized by speed/capacity
                    rem_work = sum(max(0.0, ft - t) for ft in onb) / max(1, spec["cap"])
                    # Small load-balancing tie-breaker when both analyzers are empty
                    score = rem_work + (m_jobs_count[m_idx] * 1e-4)
                    if score < best_work:
                        best_work = score
                        best_m = m_idx
        return best_m

    def schedule_wakeup(g: int, wt: float):
        if wt < pending_wakeup[g] - 1e-9:
            pending_wakeup[g] = wt
            heapq.heappush(ev_heap, (wt, EVT_TRY_ANALYZER, g, -1))

    def try_dispatch_group(g: int, t: float):
        nonlocal shared_c501_gate_free, job_wait_sum
        while waiting_jobs[g]:
            m_idx = select_analyzer_for_group(g, t)
            if m_idx is None:
                # Schedule retry at earliest next availability among analyzers in this group
                earliest_m_t = 1e18
                for c_idx in group_analyzers[g]:
                    spec = ANALYZER_SPECS[c_idx]
                    cand_t = next_launch_ready[c_idx]
                    if spec["mode"] == "seq" and len(onboard_finishes[c_idx]) >= spec["cap"]:
                        cand_t = max(cand_t, onboard_finishes[c_idx][0])
                    if spec["mode"] == "batch" and batch_busy[c_idx]:
                        continue  # EVT_BATCH_DONE will wake dispatch
                    if cand_t < earliest_m_t and cand_t > t + 1e-9:
                        earliest_m_t = cand_t
                if earliest_m_t < 1e17:
                    schedule_wakeup(g, earliest_m_t)
                return

            # Check shared C501/E601 conveyor gate if stress test active
            spec = ANALYZER_SPECS[m_idx]
            if cfg.shared_c501_e601_gate and spec["id"] in (210, 211):
                if shared_c501_gate_free > t + 1e-9:
                    schedule_wakeup(g, shared_c501_gate_free)
                    return

            # Check loader availability (cap_loaders servers)
            earliest_loader = loader_heap[0]
            if earliest_loader > t + 1e-9:
                schedule_wakeup(g, earliest_loader)
                return

            # Claim 1 loader for load_dur (1 min)
            heapq.heapreplace(loader_heap, t + load_dur)
            if cfg.shared_c501_e601_gate and spec["id"] in (210, 211):
                shared_c501_gate_free = t + 1.0

            if spec["mode"] == "batch":
                # Coagulation batch: admit up to spec['cap'] waiting coagulation jobs
                admitted = []
                while waiting_jobs[g] and len(admitted) < spec["cap"]:
                    admitted.append(pick_best_waiting_job(g, t))
                batch_busy[m_idx] = True
                start_t = t + load_dur
                max_fin = start_t
                for j in admitted:
                    w_j = t - job_rho[j]
                    if w_j > 0.0:
                        job_wait_sum += w_j
                    c_j = start_t + real_durs_list[j]
                    if c_j > max_fin:
                        max_fin = c_j
                    m_jobs_count[m_idx] += 1
                    heapq.heappush(ev_heap, (c_j, EVT_JOB_DONE_SEQ, j, -1))  # -1 = no onboard pop
                if record_traces:
                    m_busy_intervals[m_idx].append((start_t, max_fin))
                ep = _diurnal_epoch(t)
                cad = cad_epochs[str(g)][ep] * cad_mult
                next_launch_ready[m_idx] = max_fin + cad
                heapq.heappush(ev_heap, (max_fin, EVT_BATCH_DONE, g, m_idx))
            else:
                j = pick_best_waiting_job(g, t)
                w_j = t - job_rho[j]
                if w_j > 0.0:
                    job_wait_sum += w_j
                start_t = t + load_dur
                c_j = start_t + real_durs_list[j]
                heapq.heappush(onboard_finishes[m_idx], c_j)
                m_jobs_count[m_idx] += 1
                if record_traces:
                    m_busy_intervals[m_idx].append((start_t, c_j))
                ep = _diurnal_epoch(t)
                cad = cad_epochs[str(g)][ep] * cad_mult
                next_launch_ready[m_idx] = t + cad
                heapq.heappush(ev_heap, (c_j, EVT_JOB_DONE_SEQ, j, m_idx))

    # Main DES event loop
    while ev_heap:
        t, etype, arg1, arg2 = heapq.heappop(ev_heap)
        if etype == EVT_JOB_READY:
            j, g = arg1, arg2
            waiting_jobs[g].append(j)
            try_dispatch_group(g, t)
        elif etype == EVT_TRY_ANALYZER:
            if t >= pending_wakeup[arg1] - 1e-9:
                pending_wakeup[arg1] = 1e18
            try_dispatch_group(arg1, t)
        elif etype == EVT_JOB_DONE_SEQ:
            j, m_idx = arg1, arg2
            g_j = job_grp[j]
            grp_t2_t4_sum[g_j] += (t - job_t2[j])
            grp_t2_t4_cnt[g_j] += 1
            if m_idx >= 0 and onboard_finishes[m_idx]:
                # Remove finished slot from analyzer m_idx
                # Because heap pops in order, smallest element in onboard_finishes[m_idx] is typically this completion
                heapq.heappop(onboard_finishes[m_idx])
                g_m = ANALYZER_SPECS[m_idx]["group"]
                if waiting_jobs[g_m]:
                    try_dispatch_group(g_m, t)

            # Update Fork-Join encounter status & dynamic remaining work
            e = int(job_enc[j])
            enc_rem[e] -= 1
            if enc_rem[e] > 0:
                plist = enc_pred_lists[e]
                try:
                    plist.remove(float(job_pred[j]))
                except ValueError:
                    pass
                enc_max_pred_rem[e] = max(plist) if plist else 0.0
            else:
                enc_max_pred_rem[e] = 0.0
                # Fork-Join barrier satisfied: enter FCFS validation staff queue
                v_earliest = val_heap[0]
                v_start = t if t >= v_earliest else v_earliest
                v_end = v_start + q_val
                heapq.heapreplace(val_heap, v_end)
                f_e = v_end + enc_res_lag[e] * lag_mult
                tat_e = f_e - enc_first_t2[e]
                enc_tats[e] = tat_e if tat_e >= 5.0 else 5.0
        elif etype == EVT_BATCH_DONE:
            g, m_idx = arg1, arg2
            batch_busy[m_idx] = False
            if waiting_jobs[g]:
                try_dispatch_group(g, t)

    # Compute summary metrics for this day (Eq. 11)
    tats_arr = np.asarray(enc_tats, dtype=float)
    mean_tat = float(np.mean(tats_arr))
    median_tat = float(np.median(tats_arr))
    p90_tat = float(np.quantile(tats_arr, 0.90))
    sla_rate = float(np.mean(tats_arr > 120.0))
    var_tat = float(np.var(tats_arr, ddof=0))

    analyzer_jobs = {}
    analyzer_busy_min = {}
    if record_traces:
        for m_idx in active_m_indices:
            name = ANALYZER_SPECS[m_idx]["name"]
            analyzer_jobs[name] = m_jobs_count[m_idx]
            # Union of active intervals on analyzer m_idx (busy-clock minutes)
            ivs = m_busy_intervals[m_idx]
            if not ivs:
                analyzer_busy_min[name] = 0.0
            else:
                ivs.sort()
                tot_busy = 0.0
                cur_s, cur_e = ivs[0]
                for s, e_t in ivs[1:]:
                    if s <= cur_e:
                        if e_t > cur_e:
                            cur_e = e_t
                    else:
                        tot_busy += (cur_e - cur_s)
                        cur_s, cur_e = s, e_t
                tot_busy += (cur_e - cur_s)
                analyzer_busy_min[name] = tot_busy

    tot_t2_t4_sum = sum(grp_t2_t4_sum)
    tot_t2_t4_cnt = sum(grp_t2_t4_cnt)
    return DaySimOutput(
        tats=tats_arr,
        mean_tat=mean_tat,
        median_tat=median_tat,
        p90_tat=p90_tat,
        sla_rate=sla_rate,
        var_tat=var_tat,
        group_mean_t2_t4=[round(grp_t2_t4_sum[g] / max(1, grp_t2_t4_cnt[g]), 2) for g in range(5)],
        group_t2_t4_sum=grp_t2_t4_sum,
        group_t2_t4_cnt=grp_t2_t4_cnt,
        mean_job_t2_t4=tot_t2_t4_sum / max(1, tot_t2_t4_cnt),
        mean_job_wait=job_wait_sum / max(1, n_jobs),
        job_t2_t4_sum=tot_t2_t4_sum,
        job_wait_sum=job_wait_sum,
        job_count_done=tot_t2_t4_cnt,
        analyzer_jobs=analyzer_jobs,
        analyzer_busy_min=analyzer_busy_min,
    )
