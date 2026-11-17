"""
Diagnostic Analysis 4: Physical Laboratory Scan-In (RECEIVED in DT_SAMPLE_TRACKING_EVENTS),
May 11 Batch-Finalization Exclusion, Empirical T3–T4 Peak Onboard Occupancy (Filtered <=120m and Unfiltered),
and Tight Onboard & Coagulation Batch Capacity (c_m <= 10 and c_m <= 6, plus Re-Optimization under c_m <= 10).
"""

import os
import sys
import copy
import json
import numpy as np
import pandas as pd

ALG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "algorithm")
sys.path.insert(0, ALG_DIR)

import des_engine as des
from des_engine import SimConfig
from importlib.machinery import SourceFileLoader

calib_mod = SourceFileLoader("calib_mod", os.path.join(ALG_DIR, "02_calibrate_twin_parameters.py")).load_module()
opt_mod = SourceFileLoader("opt_mod", os.path.join(ALG_DIR, "03_optimize_policies.py")).load_module()
eval_mod = SourceFileLoader("eval_mod", os.path.join(ALG_DIR, "04_evaluate_heldout_and_stress.py")).load_module()


def audit_empirical_onboard_peaks(inhouse: pd.DataFrame):
    """Compute filtered (0 < T4 - T3 <= 120m) and unfiltered (T3 < T4) peak simultaneous tubes on board
    across the 6 two-way biochemistry and immunoassay analyzers that log T3."""
    valid = inhouse[inhouse["T3"].notna() & inhouse["T4"].notna() & (inhouse["T4"] > inhouse["T3"])].copy()
    valid["dur_m"] = (valid["T4"] - valid["T3"]).dt.total_seconds() / 60.0
    for label, sub in [("Filtered (0 < T4-T3 <= 120m)", valid[valid["dur_m"] <= 120.0]),
                       ("Unfiltered (all T3 < T4)", valid)]:
        peaks = {}
        for dev, grp in sub.groupby("Analyzer_ID"):
            t3s = grp["T3"].values.astype("datetime64[s]").astype(np.int64)
            t4s = grp["T4"].values.astype("datetime64[s]").astype(np.int64)
            ev_t = np.concatenate([t3s, t4s])
            ev_d = np.concatenate([np.ones_like(t3s), -np.ones_like(t4s)])
            order = np.lexsort((ev_d, ev_t))
            occ = np.cumsum(ev_d[order])
            peaks[dev] = (len(grp), int(occ.max()))
        print(f"[Empirical T3-T4 Peak Onboard - {label}] n={len(sub)}: {peaks}")


def audit_empirical_onboard_and_tight_capacity():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calib_path = os.path.join(base_dir, "outputs", "calibrated_params.json")
    weights_path = os.path.join(base_dir, "outputs", "learned_weights.json")
    if not (os.path.exists(calib_path) and os.path.exists(weights_path)):
        print("Calibration and learned weight files required in outputs/.")
        return

    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    with open(weights_path, "r", encoding="utf-8") as f:
        learned = json.load(f)

    _, inhouse = calib_mod.load_filtered_inhouse_orders()
    audit_empirical_onboard_peaks(inhouse)

    ws = calib_mod.build_daily_workloads(inhouse, calib, t4t5_cap=180.0)
    days_train = [ws[d] for d in calib["train_days_wave1"]]
    days_may = [ws[d] for d in sorted(ws.keys()) if ws[d].year_month == "2026-05"]
    days_may_no11 = [ws[d] for d in sorted(ws.keys()) if ws[d].year_month == "2026-05" and d != "2026-05-11"]
    cfg_w1 = SimConfig(prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
                       val_service_min=calib["val_service_min"], cadence_mult=calib["wave1_cadence_mult"], routing="lwr")
    w1_mem = np.array(learned["Memetic_DT_GA"]["w1_median_seed_vec"], dtype=float)

    rf_no11 = eval_mod.eval_days_detailed(days_may_no11, cfg_w1, "FCFS")
    rm_no11 = eval_mod.eval_days_detailed(days_may_no11, cfg_w1, "MEMETIC", w1_mem)
    print(f"[May excl. May 11 (30d)] FCFS: {rf_no11['pooled_mean']}m / {rf_no11['pooled_sla_pct']}% ({rf_no11['pooled_sla_cnt']}) "
          f"| Memetic: {rm_no11['pooled_mean']}m / {rm_no11['pooled_sla_pct']}% ({rm_no11['pooled_sla_cnt']})")

    orig_specs = copy.deepcopy(des.ANALYZER_SPECS)
    for cap_lim in (10, 6):
        for i, s in enumerate(des.ANALYZER_SPECS):
            s["cap"] = min(orig_specs[i]["cap"], cap_lim)
        rf = eval_mod.eval_days_detailed(days_may, cfg_w1, "FCFS")
        rs = eval_mod.eval_days_detailed(days_may, cfg_w1, "SPT")
        rm = eval_mod.eval_days_detailed(days_may, cfg_w1, "MEMETIC", w1_mem)
        print(f"[Tight Onboard & Batch c_m <= {cap_lim}] FCFS Job T2->T4={rf['job_mean_t2_t4']}m (Enc {rf['pooled_mean']}m, SLA {rf['pooled_sla_pct']}%) | "
              f"SPT Job T2->T4={rs['job_mean_t2_t4']}m (Enc {rs['pooled_mean']}m, p90 {rs['pooled_p90']}m, SLA {rs['pooled_sla_pct']}%) | "
              f"Memetic (baseline weights) Job T2->T4={rm['job_mean_t2_t4']}m (Enc {rm['pooled_mean']}m, SLA {rm['pooled_sla_pct']}%)")

        if cap_lim == 10:
            fcfs_tr_loss = opt_mod.evaluate_candidate(days_train, cfg_w1, "FCFS", None)
            spt_tr_loss = opt_mod.evaluate_candidate(days_train, cfg_w1, "SPT", None)
            fcfs_may_loss = opt_mod.evaluate_candidate(days_may, cfg_w1, "FCFS", None)
            spt_may_loss = opt_mod.evaluate_candidate(days_may, cfg_w1, "SPT", None)
            print(f"[Eq. 10 Objective under c_m <= 10] Train: FCFS={fcfs_tr_loss:.2f}, SPT={spt_tr_loss:.2f} | "
                  f"May: FCFS={fcfs_may_loss:.2f}, SPT={spt_may_loss:.2f}")

            seed_results = []
            for s_idx in range(6):
                seed = 1000 + 7 * s_idx
                w_best, l_best = opt_mod.run_memetic_ga(days_train, cfg_w1, seed=seed, n_dim=6, include_var_penalty=True)
                rm_reopt = eval_mod.eval_days_detailed(days_may, cfg_w1, "MEMETIC", w_best)
                seed_results.append((seed, l_best, rm_reopt["pooled_mean"], rm_reopt["pooled_sla_pct"]))
            seed_results.sort(key=lambda x: x[1])
            print(f"[Re-optimized Memetic DT-GA under c_m <= 10 (6 seeds, sorted by train loss)]: {seed_results}")

    for i, s in enumerate(orig_specs):
        des.ANALYZER_SPECS[i]["cap"] = s["cap"]


if __name__ == "__main__":
    audit_empirical_onboard_and_tight_capacity()
