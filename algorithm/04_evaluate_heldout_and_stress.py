"""
Step 3: Frozen Held-Out Replay, Ablations, Uncertainty & Resource-Stress Suite.
Loads strictly out-of-sample calibrated parameters and learned weights,
and executes all policies across:
  - Calibration / Training Windows (Jan-Feb 59d, Mar-Apr 61d, Jun 30d) for Table 5 fidelity
  - Held-Out Wave 1 (May 2026, 31 days) & Wave 2 (Jul-Oct 2026, 123 days) = 154 test days
  - Classical policies: FCFS, SPT, EDD, LST, HRRN, Completion-Aware Heuristic
  - Metaheuristics (6 seeds + median): Memetic DT-GA, Memetic NoVar, DE, RS, Std Linear DT-GA
  - Mechanism ablations (Table 8)
  - Weight perturbations (+/-10%, +/-20%) & T4-T5 Cap sensitivity (120m, 240m, Uncapped) (Table 9)
  - 30-replicate counterfactual T3-T4 duration resampling + 5 hospital stress scenarios (Table 10)
  - Per-analyzer job admissions & busy-clock hours: Fixed-Scan vs LWR vs Empirical Device_ID (Table 11)
"""

import concurrent.futures
import json
import os
import sys
import time
import zlib
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import stats
import openpyxl

from des_engine import ANALYZER_SPECS, DayWorkload, SimConfig, simulate_day
import importlib
cal_mod = importlib.import_module("02_calibrate_twin_parameters")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CALIB_JSON = os.path.join(OUT_DIR, "calibrated_params.json")
WEIGHTS_JSON = os.path.join(OUT_DIR, "learned_weights.json")

DAILY_RESULTS_CSV = os.path.join(OUT_DIR, "daily_simulation_results.csv")
POOLED_SUMMARY_JSON = os.path.join(OUT_DIR, "full_benchmark_outcomes.json")


def eval_days_detailed(days: List[DayWorkload], cfg: SimConfig, policy: str, w: np.ndarray = None,
                       unorm=True, usyn=True, uage=True, seed=42, return_pooled: bool = False) -> dict:
    """Run DES over a list of days using deterministic CRC32 day seeds and return daily, pooled, and job-level stats."""
    all_tats = []
    d_mean, d_med, d_p90, d_sla, d_n = [], [], [], [], []
    tot_job_t2_t4 = 0.0
    tot_job_wait = 0.0
    tot_jobs_done = 0
    tot_jobs_all = 0
    grp_sums = [0.0] * 5
    grp_cnts = [0] * 5
    for d in days:
        day_seed = seed + (zlib.crc32(d.order_date.encode("utf-8")) % 10000)
        out = simulate_day(
            d, cfg, policy=policy, w=w,
            use_normalized=unorm, use_synergy=usyn, use_quadratic_aging=uage,
            seed=day_seed
        )
        all_tats.append(out.tats)
        d_mean.append(out.mean_tat)
        d_med.append(out.median_tat)
        d_p90.append(out.p90_tat)
        d_sla.append(out.sla_rate * 100.0)
        d_n.append(d.n_encounters)
        tot_job_t2_t4 += out.job_t2_t4_sum
        tot_job_wait += out.job_wait_sum
        tot_jobs_done += out.job_count_done
        tot_jobs_all += d.n_jobs
        for g in range(5):
            grp_sums[g] += out.group_t2_t4_sum[g]
            grp_cnts[g] += out.group_t2_t4_cnt[g]

    pooled = np.concatenate(all_tats)
    res = {
        "n_days": len(days),
        "n_enc": int(len(pooled)),
        "pooled_mean": round(float(np.mean(pooled)), 2),
        "pooled_sd": round(float(np.std(pooled, ddof=1)), 2),
        "pooled_median": round(float(np.median(pooled)), 2),
        "pooled_p90": round(float(np.percentile(pooled, 90)), 2),
        "pooled_p95": round(float(np.percentile(pooled, 95)), 2),
        "pooled_sla_pct": round(float(np.mean(pooled > 120.0) * 100.0), 2),
        "pooled_sla_cnt": int(np.sum(pooled > 120.0)),
        "job_mean_t2_t4": round(tot_job_t2_t4 / max(1, tot_jobs_done), 3),
        "job_mean_wait": round(tot_job_wait / max(1, tot_jobs_all), 3),
        "grp_mean_t2_t4": [round(grp_sums[g] / max(1, grp_cnts[g]), 2) for g in range(5)],
        "daily_means": d_mean,
        "daily_p90s": d_p90,
        "daily_slas": d_sla,
        "daily_n": d_n,
    }
    if return_pooled:
        res["pooled_tats"] = pooled
    return res


def _eval_policy_bundle(args):
    """Worker to evaluate one policy across both held-out waves + across all seeds."""
    label, pmode, cw1, cw2, lkey, unorm, usyn, uage, days_may, days_jo, seed_vecs = args
    w1_vec = np.array(seed_vecs["w1_med"], dtype=float) if lkey else None
    w2_vec = np.array(seed_vecs["w2_med"], dtype=float) if lkey else None
    r_may = eval_days_detailed(days_may, cw1, pmode, w1_vec, unorm, usyn, uage)
    r_jo = eval_days_detailed(days_jo, cw2, pmode, w2_vec, unorm, usyn, uage)

    seed_details = []
    if lkey:
        for s_idx, (sv1, sv2) in enumerate(zip(seed_vecs["w1_all"], seed_vecs["w2_all"])):
            sm = eval_days_detailed(days_may, cw1, pmode, np.array(sv1), unorm, usyn, uage)
            sj = eval_days_detailed(days_jo, cw2, pmode, np.array(sv2), unorm, usyn, uage)
            seed_details.append({
                "seed_idx": s_idx,
                "may_mean": sm["pooled_mean"], "may_p90": sm["pooled_p90"], "may_sla_pct": sm["pooled_sla_pct"],
                "jo_mean": sj["pooled_mean"], "jo_p90": sj["pooled_p90"], "jo_sla_pct": sj["pooled_sla_pct"],
            })

    return label, r_may, r_jo, seed_details


def _run_resamp_replicate(args):
    rep, days_may, days_jo, pools, cfg_w1_lwr, cfg_w2_lwr, w1_mem, w2_mem = args
    rng_r = np.random.default_rng(7000 + rep)
    resamp_may = [
        DayWorkload(
            d.order_date, d.year_month, d.n_jobs, d.n_encounters, d.job_enc, d.job_group,
            d.job_t2, d.job_pred_dur,
            np.array([pools[g][int(rng_r.integers(0, len(pools[g])))] for g in d.job_group], dtype=float),
            d.enc_first_t2, d.enc_job_counts, d.enc_res_lag, d.obs_tat_cap180, d.obs_tat_uncap
        ) for d in days_may
    ]
    resamp_jo = [
        DayWorkload(
            d.order_date, d.year_month, d.n_jobs, d.n_encounters, d.job_enc, d.job_group,
            d.job_t2, d.job_pred_dur,
            np.array([pools[g][int(rng_r.integers(0, len(pools[g])))] for g in d.job_group], dtype=float),
            d.enc_first_t2, d.enc_job_counts, d.enc_res_lag, d.obs_tat_cap180, d.obs_tat_uncap
        ) for d in days_jo
    ]
    rfm = eval_days_detailed(resamp_may, cfg_w1_lwr, "FCFS")
    rmm = eval_days_detailed(resamp_may, cfg_w1_lwr, "MEMETIC", w1_mem)
    rfj = eval_days_detailed(resamp_jo, cfg_w2_lwr, "FCFS")
    rmj = eval_days_detailed(resamp_jo, cfg_w2_lwr, "MEMETIC", w2_mem)
    return rfm, rmm, rfj, rmj


def main():
    t0 = time.time()
    with open(CALIB_JSON, "r", encoding="utf-8") as f:
        calib = json.load(f)
    with open(WEIGHTS_JSON, "r", encoding="utf-8") as f:
        learned = json.load(f)

    raw_df_full, inhouse = cal_mod.load_filtered_inhouse_orders()
    print("[+] Building 304-day exact workloads (180-min T4-T5 primary cap)...")
    ws_cap180 = cal_mod.build_daily_workloads(inhouse, calib, t4t5_cap=180.0)

    def get_window(months):
        return [ws_cap180[d] for d in sorted(ws_cap180.keys()) if ws_cap180[d].year_month in months]

    days_jf = get_window(["2026-01", "2026-02"])
    days_ma = get_window(["2026-03", "2026-04"])
    days_may = get_window(["2026-05"])
    days_jun = get_window(["2026-06"])
    days_jo = get_window(["2026-07", "2026-08", "2026-09", "2026-10"])
    days_held = days_may + days_jo

    cfg_w1_lwr = SimConfig(prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
                           val_service_min=calib["val_service_min"], cadence_mult=calib["wave1_cadence_mult"], routing="lwr")
    cfg_w2_lwr = SimConfig(prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
                           val_service_min=calib["val_service_min"], cadence_mult=calib["wave2_cadence_mult"], routing="lwr")
    cfg_w1_fix = SimConfig(prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
                           val_service_min=calib["val_service_min"], cadence_mult=calib["wave1_cadence_mult"], routing="fixed")
    cfg_w2_fix = SimConfig(prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
                           val_service_min=calib["val_service_min"], cadence_mult=calib["wave2_cadence_mult"], routing="fixed")

    # ------------------------------------------------------------------
    # 1. TABLE 5: Calibration & Held-Out FCFS Fidelity vs Observed Source
    # ------------------------------------------------------------------
    print("[+] Computing Table 5 fidelity across all 5 epochs...")
    table5_rows = []
    epoch_specs = [
        ("Wave1_Cal_JanFeb", days_jf, cfg_w1_lwr),
        ("Wave1_Train_MarApr", days_ma, cfg_w1_lwr),
        ("Wave1_Test_May", days_may, cfg_w1_lwr),
        ("Wave2_Recalib_Jun", days_jun, cfg_w2_lwr),
        ("Wave2_Valid_JulOct", days_jo, cfg_w2_lwr),
    ]
    def tost_p(diff_arr: np.ndarray, margin: float) -> float:
        p1 = float(stats.ttest_1samp(diff_arr, -margin, alternative="greater").pvalue)
        p2 = float(stats.ttest_1samp(diff_arr, margin, alternative="less").pvalue)
        return round(max(p1, p2), 4)

    for ep_name, ep_days, ep_cfg in epoch_specs:
        fcfs_res = eval_days_detailed(ep_days, ep_cfg, "FCFS", return_pooled=True)
        obs_c180 = np.concatenate([d.obs_tat_cap180 for d in ep_days if len(d.obs_tat_cap180) > 0])
        obs_unc = np.concatenate([d.obs_tat_uncap for d in ep_days if len(d.obs_tat_uncap) > 0])
        src_dm = np.array([float(np.mean(d.obs_tat_cap180)) for d in ep_days if len(d.obs_tat_cap180) > 0])
        sim_dm = np.array([m for d, m in zip(ep_days, fcfs_res["daily_means"]) if len(d.obs_tat_cap180) > 0])
        src_dp90 = np.array([float(np.percentile(d.obs_tat_cap180, 90)) for d in ep_days if len(d.obs_tat_cap180) > 0])
        sim_dp90 = np.array([p for d, p in zip(ep_days, fcfs_res["daily_p90s"]) if len(d.obs_tat_cap180) > 0])
        ks_stat, ks_p = stats.ks_2samp(fcfs_res["pooled_tats"], obs_c180)
        diff_dm = sim_dm - src_dm
        diff_dp90 = sim_dp90 - src_dp90
        table5_rows.append({
            "epoch": ep_name,
            "days": len(ep_days),
            "encounters_modeled": fcfs_res["n_enc"],
            "encounters_valid_t5": int(len(obs_c180)),
            "src_cap180_mean": round(float(np.mean(obs_c180)), 2),
            "src_cap180_med": round(float(np.median(obs_c180)), 2),
            "src_cap180_p90": round(float(np.percentile(obs_c180, 90)), 2),
            "src_cap180_p95": round(float(np.percentile(obs_c180, 95)), 2),
            "src_cap180_sla_pct": round(float(np.mean(obs_c180 > 120.0) * 100.0), 2),
            "src_uncap_mean": round(float(np.mean(obs_unc)), 2),
            "src_uncap_sla_pct": round(float(np.mean(obs_unc > 120.0) * 100.0), 2),
            "sim_fcfs_mean": fcfs_res["pooled_mean"],
            "sim_fcfs_med": fcfs_res["pooled_median"],
            "sim_fcfs_p90": fcfs_res["pooled_p90"],
            "sim_fcfs_p95": fcfs_res["pooled_p95"],
            "sim_fcfs_sla_pct": fcfs_res["pooled_sla_pct"],
            "daily_bias_mean": round(float(np.mean(diff_dm)), 2),
            "daily_mae_mean": round(float(np.mean(np.abs(diff_dm))), 2),
            "tost_mean_p": tost_p(diff_dm, 15.0),
            "tost_p90_p": tost_p(diff_dp90, 20.0),
            "ks_stat": round(float(ks_stat), 4),
            "ks_p": float(ks_p),
        })

    # ------------------------------------------------------------------
    # 2. TABLES 6 & 7: Held-Out Evaluation on May (31d) + Jul-Oct (123d)
    # ------------------------------------------------------------------
    print("[+] Evaluating all Classical, Linear GA, DE, RS, and Memetic DT-GA policies on 154 held-out days...")
    policy_configs = [
        ("FCFS",                    "FCFS",       cfg_w1_lwr, cfg_w2_lwr, None, True,  True,  True),
        ("SPT",                     "SPT",        cfg_w1_lwr, cfg_w2_lwr, None, True,  True,  True),
        ("EDD",                     "EDD",        cfg_w1_lwr, cfg_w2_lwr, None, True,  True,  True),
        ("LST",                     "LST",        cfg_w1_lwr, cfg_w2_lwr, None, True,  True,  True),
        ("HRRN",                    "HRRN",       cfg_w1_lwr, cfg_w2_lwr, None, True,  True,  True),
        ("Completion_Heuristic",    "COMPLETION", cfg_w1_lwr, cfg_w2_lwr, None, True,  True,  True),
        ("Standard_Linear_DT_GA",   "STD_LINEAR", cfg_w1_lwr, cfg_w2_lwr, "Standard_Linear_DT_GA", False, False, False),
        ("Differential_Evolution",  "MEMETIC",    cfg_w1_lwr, cfg_w2_lwr, "Differential_Evolution", True, True, True),
        ("Random_Search",           "MEMETIC",    cfg_w1_lwr, cfg_w2_lwr, "Random_Search", True, True, True),
        ("Memetic_DT_GA_NoVar",     "MEMETIC",    cfg_w1_lwr, cfg_w2_lwr, "Memetic_DT_GA_NoVar", True, True, True),
        ("Memetic_DT_GA",           "MEMETIC",    cfg_w1_lwr, cfg_w2_lwr, "Memetic_DT_GA", True, True, True),
    ]

    daily_records = pd.DataFrame({
        "OrderDate": [d.order_date for d in days_held],
        "YearMonth": [d.year_month for d in days_held],
        "Wave": ["Wave1_May" if d.year_month == "2026-05" else "Wave2_JulOct" for d in days_held],
        "InHouse_Encounters": [d.n_encounters for d in days_held],
        "InHouse_Jobs": [d.n_jobs for d in days_held],
        "Source_Cap180_Mean_TAT": [round(float(np.mean(d.obs_tat_cap180)), 3) if len(d.obs_tat_cap180) else np.nan for d in days_held],
        "Source_Uncap_Mean_TAT": [round(float(np.mean(d.obs_tat_uncap)), 3) if len(d.obs_tat_uncap) else np.nan for d in days_held],
    })

    p_tasks = []
    for label, pmode, cw1, cw2, lkey, unorm, usyn, uage in policy_configs:
        svecs = {"w1_med": learned[lkey]["w1_median_seed_vec"], "w2_med": learned[lkey]["w2_median_seed_vec"],
                 "w1_all": learned[lkey]["w1_all_seed_vecs"], "w2_all": learned[lkey]["w2_all_seed_vecs"]} if lkey else None
        p_tasks.append((label, pmode, cw1, cw2, lkey, unorm, usyn, uage, days_may, days_jo, svecs))

    n_workers = min(16, max(1, (os.cpu_count() or 2) - 2))
    table6_rows = []
    table12_rows = []
    per_seed_table = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for label, r_may, r_jo, seed_details in ex.map(_eval_policy_bundle, p_tasks):
            daily_records[f"{label}_Mean"] = r_may["daily_means"] + r_jo["daily_means"]
            daily_records[f"{label}_p90"] = r_may["daily_p90s"] + r_jo["daily_p90s"]
            daily_records[f"{label}_SLA_Pct"] = r_may["daily_slas"] + r_jo["daily_slas"]

            sm_m = [s["may_mean"] for s in seed_details]
            sj_m = [s["jo_mean"] for s in seed_details]
            sm_p = [s["may_p90"] for s in seed_details]
            sj_p = [s["jo_p90"] for s in seed_details]
            sm_s = [s["may_sla_pct"] for s in seed_details]
            sj_s = [s["jo_sla_pct"] for s in seed_details]
            if seed_details:
                per_seed_table[label] = seed_details

            table6_rows.append({
                "policy": label,
                "may_mean": r_may["pooled_mean"], "may_sd": r_may["pooled_sd"], "may_med": r_may["pooled_median"],
                "may_p90": r_may["pooled_p90"], "may_sla_pct": r_may["pooled_sla_pct"], "may_sla_cnt": r_may["pooled_sla_cnt"],
                "jo_mean": r_jo["pooled_mean"], "jo_sd": r_jo["pooled_sd"], "jo_med": r_jo["pooled_median"],
                "jo_p90": r_jo["pooled_p90"], "jo_sla_pct": r_jo["pooled_sla_pct"], "jo_sla_cnt": r_jo["pooled_sla_cnt"],
                "seed_may_mean": round(float(np.mean(sm_m)), 2) if sm_m else None,
                "seed_may_sd": round(float(np.std(sm_m, ddof=1)), 2) if sm_m else None,
                "seed_may_p90_sd": round(float(np.std(sm_p, ddof=1)), 2) if sm_p else None,
                "seed_may_sla_sd": round(float(np.std(sm_s, ddof=1)), 2) if sm_s else None,
                "seed_jo_mean": round(float(np.mean(sj_m)), 2) if sj_m else None,
                "seed_jo_sd": round(float(np.std(sj_m, ddof=1)), 2) if sj_m else None,
                "seed_jo_p90_sd": round(float(np.std(sj_p, ddof=1)), 2) if sj_p else None,
                "seed_jo_sla_sd": round(float(np.std(sj_s, ddof=1)), 2) if sj_s else None,
            })

            table12_rows.append({
                "policy": label,
                "may_job_t2_t4": r_may["job_mean_t2_t4"],
                "may_job_wait": r_may["job_mean_wait"],
                "may_grp_t2_t4": r_may["grp_mean_t2_t4"],
                "may_enc_mean": r_may["pooled_mean"],
                "jo_job_t2_t4": r_jo["job_mean_t2_t4"],
                "jo_job_wait": r_jo["job_mean_wait"],
                "jo_grp_t2_t4": r_jo["grp_mean_t2_t4"],
                "jo_enc_mean": r_jo["pooled_mean"],
            })

    daily_records.to_csv(DAILY_RESULTS_CSV, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 3. TABLE 8: Stepwise Mechanism Ablation (Median + 6-Seed Mean +- SD, Equal LWR Routing)
    # ------------------------------------------------------------------
    print("[+] Evaluating Table 8 stepwise ablations across all 6 seeds...")
    ablation_specs = [
        ("Full_Memetic_DT_GA",     "Memetic_DT_GA",          "MEMETIC",    True,  True,  True,  cfg_w1_lwr, cfg_w2_lwr),
        ("Minus_NelderMead",       "Ablation_No_NelderMead", "MEMETIC",    True,  True,  True,  cfg_w1_lwr, cfg_w2_lwr),
        ("Minus_QuadAging",        "Ablation_No_QuadAging",  "MEMETIC",    True,  True,  False, cfg_w1_lwr, cfg_w2_lwr),
        ("Minus_ForkJoinSynergy",  "Ablation_No_Synergy",    "MEMETIC",    True,  False, True,  cfg_w1_lwr, cfg_w2_lwr),
        ("Unnormalized_Linear_LWR","Standard_Linear_DT_GA",  "STD_LINEAR", False, False, False, cfg_w1_lwr, cfg_w2_lwr),
    ]
    table8_rows = []
    for alabel, lkey, pmode, unorm, usyn, uage, cw1, cw2 in ablation_specs:
        w1_v = np.array(learned[lkey]["w1_median_seed_vec"], dtype=float)
        w2_v = np.array(learned[lkey]["w2_median_seed_vec"], dtype=float)
        rm = eval_days_detailed(days_may, cw1, pmode, w1_v, unorm, usyn, uage)
        rj = eval_days_detailed(days_jo, cw2, pmode, w2_v, unorm, usyn, uage)
        sm_m, sm_p, sm_s = [], [], []
        sj_m, sj_p, sj_s = [], [], []
        for sv1, sv2 in zip(learned[lkey]["w1_all_seed_vecs"], learned[lkey]["w2_all_seed_vecs"]):
            am = eval_days_detailed(days_may, cw1, pmode, np.array(sv1, dtype=float), unorm, usyn, uage)
            aj = eval_days_detailed(days_jo, cw2, pmode, np.array(sv2, dtype=float), unorm, usyn, uage)
            sm_m.append(am["pooled_mean"]); sm_p.append(am["pooled_p90"]); sm_s.append(am["pooled_sla_pct"])
            sj_m.append(aj["pooled_mean"]); sj_p.append(aj["pooled_p90"]); sj_s.append(aj["pooled_sla_pct"])
        table8_rows.append({
            "variant": alabel,
            "may_mean": rm["pooled_mean"], "may_p90": rm["pooled_p90"], "may_sla_pct": rm["pooled_sla_pct"],
            "jo_mean": rj["pooled_mean"], "jo_p90": rj["pooled_p90"], "jo_sla_pct": rj["pooled_sla_pct"],
            "seed_may_mean": round(float(np.mean(sm_m)), 2), "seed_may_mean_sd": round(float(np.std(sm_m, ddof=1)), 2),
            "seed_may_p90": round(float(np.mean(sm_p)), 2),  "seed_may_p90_sd": round(float(np.std(sm_p, ddof=1)), 2),
            "seed_may_sla": round(float(np.mean(sm_s)), 2),  "seed_may_sla_sd": round(float(np.std(sm_s, ddof=1)), 2),
            "seed_jo_mean": round(float(np.mean(sj_m)), 2),  "seed_jo_mean_sd": round(float(np.std(sj_m, ddof=1)), 2),
            "seed_jo_p90": round(float(np.mean(sj_p)), 2),   "seed_jo_p90_sd": round(float(np.std(sj_p, ddof=1)), 2),
            "seed_jo_sla": round(float(np.mean(sj_s)), 2),   "seed_jo_sla_sd": round(float(np.std(sj_s, ddof=1)), 2),
        })

    # ------------------------------------------------------------------
    # 4. TABLE 9: Scale Invariance, Non-Uniform Gene Perturbations, Lambda & Cap Sensitivity
    # ------------------------------------------------------------------
    print("[+] Verifying uniform scale invariance, non-uniform gene sensitivity, lambda & T4-T5 caps...")
    w1_mem = np.array(learned["Memetic_DT_GA"]["w1_median_seed_vec"], dtype=float)
    w2_mem = np.array(learned["Memetic_DT_GA"]["w2_median_seed_vec"], dtype=float)
    table9_sens = []

    # 4A. Uniform scaling test (must yield exact 0.00 min change by Eq. 1 monotonicity)
    for mult_lbl, mult in [("UniformScale_x1.10", 1.10), ("UniformScale_x0.90", 0.90), ("UniformScale_x1.20", 1.20), ("UniformScale_x0.80", 0.80)]:
        rm = eval_days_detailed(days_may, cfg_w1_lwr, "MEMETIC", w1_mem * mult)
        rj = eval_days_detailed(days_jo, cfg_w2_lwr, "MEMETIC", w2_mem * mult)
        table9_sens.append({
            "scenario": mult_lbl,
            "may_mean": rm["pooled_mean"], "may_p90": rm["pooled_p90"], "may_sla_pct": rm["pooled_sla_pct"],
            "jo_mean": rj["pooled_mean"], "jo_p90": rj["pooled_p90"], "jo_sla_pct": rj["pooled_sla_pct"],
        })

    # 4B. Non-uniform single-gene & alternating perturbations (+/-20%)
    nonunif_specs = [
        ("Gene_w1_wait_+20pct",    np.array([1.2, 1.0, 1.0, 1.0, 1.0, 1.0])),
        ("Gene_w1_wait_-20pct",    np.array([0.8, 1.0, 1.0, 1.0, 1.0, 1.0])),
        ("Gene_w3_synergy_+20pct", np.array([1.0, 1.0, 1.2, 1.0, 1.0, 1.0])),
        ("Gene_w6_aging_+20pct",   np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.2])),
        ("Gene_w6_aging_-20pct",   np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.8])),
        ("Alternating_+-20pct",    np.array([1.2, 0.8, 1.2, 0.8, 1.2, 0.8])),
    ]
    for nlbl, vec_m in nonunif_specs:
        rm = eval_days_detailed(days_may, cfg_w1_lwr, "MEMETIC", w1_mem * vec_m)
        rj = eval_days_detailed(days_jo, cfg_w2_lwr, "MEMETIC", w2_mem * vec_m)
        table9_sens.append({
            "scenario": nlbl,
            "may_mean": rm["pooled_mean"], "may_p90": rm["pooled_p90"], "may_sla_pct": rm["pooled_sla_pct"],
            "jo_mean": rj["pooled_mean"], "jo_p90": rj["pooled_p90"], "jo_sla_pct": rj["pooled_sla_pct"],
        })

    # 4C. Objective loss coefficient sensitivity (retrained with 6 seeds x 112 evals)
    for lkey in ["Lambda90_0.20", "Lambda90_0.50", "LambdaSLA_60", "LambdaSLA_120", "LambdaVar_0.50"]:
        if lkey in learned:
            rm = eval_days_detailed(days_may, cfg_w1_lwr, "MEMETIC", np.array(learned[lkey]["w1_median_seed_vec"]))
            rj = eval_days_detailed(days_jo, cfg_w2_lwr, "MEMETIC", np.array(learned[lkey]["w2_median_seed_vec"]))
            table9_sens.append({
                "scenario": lkey,
                "may_mean": rm["pooled_mean"], "may_p90": rm["pooled_p90"], "may_sla_pct": rm["pooled_sla_pct"],
                "jo_mean": rj["pooled_mean"], "jo_p90": rj["pooled_p90"], "jo_sla_pct": rj["pooled_sla_pct"],
            })

    # Build on full `inhouse` so imputed 1-way p_tilde RNG sequence matches baseline identically across cap thresholds
    for cap_lbl, cap_val in [("T4T5_Cap_120m", 120.0), ("T4T5_Cap_240m", 240.0), ("T4T5_Uncapped", None)]:
        ws_c = cal_mod.build_daily_workloads(inhouse, calib, t4t5_cap=cap_val)
        d_m = [ws_c[k] for k in sorted(ws_c) if ws_c[k].year_month == "2026-05"]
        d_j = [ws_c[k] for k in sorted(ws_c) if ws_c[k].year_month >= "2026-07"]
        f_m = eval_days_detailed(d_m, cfg_w1_lwr, "FCFS")["pooled_mean"]
        m_m = eval_days_detailed(d_m, cfg_w1_lwr, "MEMETIC", w1_mem)
        f_j = eval_days_detailed(d_j, cfg_w2_lwr, "FCFS")["pooled_mean"]
        m_j = eval_days_detailed(d_j, cfg_w2_lwr, "MEMETIC", w2_mem)
        table9_sens.append({
            "scenario": cap_lbl,
            "may_fcfs_mean": f_m, "may_mem_mean": m_m["pooled_mean"], "may_mem_p90": m_m["pooled_p90"], "may_mem_sla": m_m["pooled_sla_pct"],
            "jo_fcfs_mean": f_j, "jo_mem_mean": m_j["pooled_mean"], "jo_mem_p90": m_j["pooled_p90"], "jo_mem_sla": m_j["pooled_sla_pct"],
        })

    # ------------------------------------------------------------------
    # 5. TABLE 10: 30-Replicate Resampling & 5 Laboratory Stress Tests
    # ------------------------------------------------------------------
    print("[+] Running 30-replicate counterfactual T3-T4 duration resampling + stress scenarios...")
    pools = {int(k): np.array(v, dtype=float) for k, v in calib["t3t4_pools"].items()}
    rep_fcfs_may, rep_mem_may, rep_fcfs_jo, rep_mem_jo = [], [], [], []
    rep_sla_fcfs_may, rep_sla_mem_may, rep_sla_fcfs_jo, rep_sla_mem_jo = [], [], [], []
    rep_p90_wins_may, rep_p90_wins_jo = 0, 0

    n_workers = min(16, max(1, (os.cpu_count() or 2) - 2))
    rep_args = [(rep, days_may, days_jo, pools, cfg_w1_lwr, cfg_w2_lwr, w1_mem, w2_mem) for rep in range(30)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for rfm, rmm, rfj, rmj in ex.map(_run_resamp_replicate, rep_args):
            rep_fcfs_may.append(rfm["pooled_mean"]); rep_mem_may.append(rmm["pooled_mean"])
            rep_sla_fcfs_may.append(rfm["pooled_sla_pct"]); rep_sla_mem_may.append(rmm["pooled_sla_pct"])
            rep_fcfs_jo.append(rfj["pooled_mean"]); rep_mem_jo.append(rmj["pooled_mean"])
            rep_sla_fcfs_jo.append(rfj["pooled_sla_pct"]); rep_sla_mem_jo.append(rmj["pooled_sla_pct"])
            if rmm["pooled_p90"] < rfm["pooled_p90"]: rep_p90_wins_may += 1
            if rmj["pooled_p90"] < rfj["pooled_p90"]: rep_p90_wins_jo += 1

    table10_rows = [
        {
            "scenario": "30_Replicate_T3T4_Resampling",
            "may_fcfs_mean": round(float(np.mean(rep_fcfs_may)), 2), "may_mem_mean": round(float(np.mean(rep_mem_may)), 2),
            "may_wins_mean": int(np.sum(np.array(rep_mem_may) < np.array(rep_fcfs_may))),
            "may_wins_p90": rep_p90_wins_may,
            "may_wins_sla": int(np.sum(np.array(rep_sla_mem_may) < np.array(rep_sla_fcfs_may))),
            "jo_fcfs_mean": round(float(np.mean(rep_fcfs_jo)), 2), "jo_mem_mean": round(float(np.mean(rep_mem_jo)), 2),
            "jo_wins_mean": int(np.sum(np.array(rep_mem_jo) < np.array(rep_fcfs_jo))),
            "jo_wins_p90": rep_p90_wins_jo,
            "jo_wins_sla": int(np.sum(np.array(rep_sla_mem_jo) < np.array(rep_sla_fcfs_jo))),
        }
    ]

    def clone_cfg(base: SimConfig, **kw) -> SimConfig:
        d = {k: getattr(base, k) for k in base.__dataclass_fields__}
        d.update(kw)
        return SimConfig(**d)

    stress_defs = [
        ("Cadence_+15pct",             {"cadence_mult_rel": 1.15}),
        ("Cadence_-15pct",             {"cadence_mult_rel": 0.85}),
        ("Lag_+25pct",                 {"lag_mult": 1.25}),
        ("Shared_C501_E601_Gate",      {"shared_c501_e601_gate": True}),
        ("Advia1650_Outage",           {"outage_advia1650": True}),
        ("Loaders_8_to_6",             {"cap_loaders": 6}),
        ("Validators_5_to_3",          {"cap_validators": 3}),
        ("Analytical_Rerun_5pct_2x",   {"rerun_prob": 0.05, "rerun_mult": 2.0}),
    ]
    for s_name, s_kw in stress_defs:
        kw1, kw2 = dict(s_kw), dict(s_kw)
        rel = kw1.pop("cadence_mult_rel", None)
        if rel is not None:
            kw2.pop("cadence_mult_rel", None)
            kw1["cadence_mult"] = cfg_w1_lwr.cadence_mult * rel
            kw2["cadence_mult"] = cfg_w2_lwr.cadence_mult * rel
        sc1, sc2 = clone_cfg(cfg_w1_lwr, **kw1), clone_cfg(cfg_w2_lwr, **kw2)
        fm, mm = eval_days_detailed(days_may, sc1, "FCFS"), eval_days_detailed(days_may, sc1, "MEMETIC", w1_mem)
        fj, mj = eval_days_detailed(days_jo, sc2, "FCFS"), eval_days_detailed(days_jo, sc2, "MEMETIC", w2_mem)
        table10_rows.append({
            "scenario": s_name,
            "may_fcfs_mean": fm["pooled_mean"], "may_mem_mean": mm["pooled_mean"],
            "may_delta_mean": round(mm["pooled_mean"] - fm["pooled_mean"], 2),
            "may_fcfs_sla": fm["pooled_sla_pct"], "may_mem_sla": mm["pooled_sla_pct"],
            "jo_fcfs_mean": fj["pooled_mean"], "jo_mem_mean": mj["pooled_mean"],
            "jo_delta_mean": round(mj["pooled_mean"] - fj["pooled_mean"], 2),
            "jo_fcfs_sla": fj["pooled_sla_pct"], "jo_mem_sla": mj["pooled_sla_pct"],
        })

    # ------------------------------------------------------------------
    # 6. TABLE 11: Per-Analyzer Simulated Jobs & Busy-Clock Hours Audit
    # ------------------------------------------------------------------
    print("[+] Auditing Table 11 per-analyzer utilization (Fixed vs LWR vs Empirical Device_ID)...")
    def aggregate_analyzer_traces(days_list, cfg, pmode, w_vec):
        jobs_tot = {s["name"]: 0 for s in ANALYZER_SPECS}
        busy_tot = {s["name"]: 0.0 for s in ANALYZER_SPECS}
        for d in days_list:
            out = simulate_day(d, cfg, policy=pmode, w=w_vec, record_traces=True)
            for k, v in out.analyzer_jobs.items():
                jobs_tot[k] += v
            for k, v in out.analyzer_busy_min.items():
                busy_tot[k] += v / 60.0
        return jobs_tot, {k: round(v, 1) for k, v in busy_tot.items()}

    fix_may_j, fix_may_h = aggregate_analyzer_traces(days_may, cfg_w1_fix, "FCFS", None)
    lwr_may_j, lwr_may_h = aggregate_analyzer_traces(days_may, cfg_w1_lwr, "MEMETIC", w1_mem)
    lwr_jo_j, lwr_jo_h = aggregate_analyzer_traces(days_jo, cfg_w2_lwr, "MEMETIC", w2_mem)

    table11_rows = []
    for s in ANALYZER_SPECS:
        nm = s["name"]
        table11_rows.append({
            "module": nm, "device_id": s["id"],
            "fixed_may_jobs": fix_may_j[nm], "fixed_may_busy_h": fix_may_h[nm],
            "lwr_may_jobs": lwr_may_j[nm], "lwr_may_busy_h": lwr_may_h[nm],
            "lwr_jo_jobs": lwr_jo_j[nm], "lwr_jo_busy_h": lwr_jo_h[nm],
        })

    # Additional mechanism & cohort audits:
    # (a) Cross-test Wave 2 median chromosome on congested May (proves age-monotone FCFS rank equivalence)
    w2_on_may_identical_days = 0
    for d in days_may:
        rf = simulate_day(d, cfg_w1_lwr, "FCFS")
        rw2 = simulate_day(d, cfg_w1_lwr, "MEMETIC", w2_mem)
        if np.allclose(rf.tats, rw2.tats, atol=1e-9):
            w2_on_may_identical_days += 1

    # (b) Exploratory May branch-count decomposition (FCFS vs Memetic vs zero-(w2,w3) age-only ablation)
    w1_age_only = np.array([w1_mem[0], 0.0, 0.0, 0.0, w1_mem[4], w1_mem[5]], dtype=float)
    fc_all, mm_all, a0_all, cnt_all = [], [], [], []
    for d in days_may:
        fc_all.append(simulate_day(d, cfg_w1_lwr, "FCFS").tats)
        mm_all.append(simulate_day(d, cfg_w1_lwr, "MEMETIC", w1_mem).tats)
        a0_all.append(simulate_day(d, cfg_w1_lwr, "MEMETIC", w1_age_only).tats)
        cnt_all.append(d.enc_job_counts)
    fc_all = np.concatenate(fc_all); mm_all = np.concatenate(mm_all); a0_all = np.concatenate(a0_all); cnt_all = np.concatenate(cnt_all)
    branch_decomp = {}
    for bk, blbl in [(1, "1_branch"), (2, "2_branch"), (3, "3_branch"), (4, "4plus_branch")]:
        msk = (cnt_all == bk) if bk < 4 else (cnt_all >= 4)
        fb = int(np.sum(fc_all[msk] > 120.0))
        mb = int(np.sum(mm_all[msk] > 120.0))
        ab = int(np.sum(a0_all[msk] > 120.0))
        branch_decomp[blbl] = {"n_enc": int(np.sum(msk)), "fcfs_breaches": fb, "memetic_breaches": mb, "delta_mem_vs_fcfs": mb - fb, "age_only_breaches": ab}

    # (c) Exact send-out encounter-day classification breakdown
    raw_df_full["DateEnc"] = raw_df_full["OrderDate"].astype(str) + "_" + raw_df_full["EncounterKey"].astype(str)
    inhouse["DateEnc"] = inhouse["OrderDate"].astype(str) + "_" + inhouse["EncounterKey"].astype(str)
    ih_any_df = raw_df_full[raw_df_full["IsExternalOrder"] == 0]
    ext_enc_s = set(raw_df_full.loc[raw_df_full["IsExternalOrder"] == 1, "DateEnc"].unique())
    mod_enc_s = set(inhouse["DateEnc"].unique())
    any_ih_s = set((ih_any_df["OrderDate"].astype(str) + "_" + ih_any_df["EncounterKey"].astype(str)).unique())
    no_mod_s = ext_enc_s - mod_enc_s
    sendout_enc_breakdown = {
        "total_encounters_with_sendout": len(ext_enc_s),
        "mixed_with_modeled_inhouse": len(ext_enc_s & mod_enc_s),
        "without_modeled_automated_inhouse": len(no_mod_s),
        "truly_external_only": len(ext_enc_s - any_ih_s),
        "external_plus_unmodeled_manual_inhouse": len(no_mod_s & any_ih_s),
    }

    bundle_out = {
        "runtime_sec": round(time.time() - t0, 2),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": stats.scipy.__version__ if hasattr(stats, "scipy") else __import__("scipy").__version__,
            "openpyxl": openpyxl.__version__,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "unset"),
        },
        "calibrated_wave1_scale": calib["wave1_cadence_mult"],
        "calibrated_wave2_scale": calib["wave2_cadence_mult"],
        "learned_weights": {k: {"W1_May": v["w1_median_seed_vec"], "W2_JulOct": v["w2_median_seed_vec"]} for k, v in learned.items()},
        "per_seed_outcomes": per_seed_table,
        "wave2_fcfs_equivalence_audit": {
            "w2_median_weights": [round(float(x), 4) for x in w2_mem],
            "w2_on_may_identical_days": f"{w2_on_may_identical_days}/{len(days_may)}",
        },
        "may_branch_decomposition_audit": branch_decomp,
        "sendout_encounter_breakdown": sendout_enc_breakdown,
        "table5_fidelity": table5_rows,
        "table6_pooled_performance": table6_rows,
        "table8_ablations": table8_rows,
        "table9_sensitivity": table9_sens,
        "table10_resampling_and_stress": table10_rows,
        "table11_device_traces": table11_rows,
        "table12_kleinrock_conservation": table12_rows,
    }
    with open(POOLED_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(bundle_out, f, indent=2)
    print(f"[DONE] Completed entire held-out, stress, ablation, and device audit suite in {time.time() - t0:.2f}s -> {POOLED_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
