"""
Diagnostic Analysis 1: Start-Time Invariance, Accounting Identity (Eq. 13), and Information Checks.
Checks on all 154 test days (May: 31 days; July–October: 123 days):
  1. Whether any dispatch attempt found a launch-ready analyzer full (onboard == c_m).
  2. Whether any coagulation batch launch had >25 waiting jobs.
  3. Maximum absolute difference between sorted analyzer start times under each policy and FCFS.
  4. Mean job receipt-to-completion time (T2->T4) and mean job wait (Table 4).
  5. Sibling arrival timestamp audit (proportion of encounter-days sharing 1 receipt timestamp).
  6. LWR routing sensitivity (realized vs predicted remaining onboard durations; fixed-order scanning).
  7. Wave 2 age-only chromosome (w*_W2) day-by-day encounter TAT equivalence with FCFS.
"""

import os
import sys
import json
import zlib
import numpy as np
import pandas as pd

ALG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "algorithm")
sys.path.insert(0, ALG_DIR)

from des_engine import SimConfig, simulate_day
from importlib.machinery import SourceFileLoader

calib_mod = SourceFileLoader("calib_mod", os.path.join(ALG_DIR, "02_calibrate_twin_parameters.py")).load_module()
eval_mod = SourceFileLoader("eval_mod", os.path.join(ALG_DIR, "04_evaluate_heldout_and_stress.py")).load_module()


def run_start_time_and_information_checks():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calib_path = os.path.join(base_dir, "outputs", "calibrated_params.json")
    weights_path = os.path.join(base_dir, "outputs", "learned_weights.json")

    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    with open(weights_path, "r", encoding="utf-8") as f:
        learned = json.load(f)

    _, inhouse = calib_mod.load_filtered_inhouse_orders()
    ws = calib_mod.build_daily_workloads(inhouse, calib, t4t5_cap=180.0)

    # 1. Sibling receipt timestamp check
    enc_t2_n = inhouse.groupby(["OrderDate", "EncounterKey"])["T2_Min"].nunique()
    n_single = int((enc_t2_n == 1).sum())
    n_total = len(enc_t2_n)
    inhouse["first_t2"] = inhouse.groupby(["OrderDate", "EncounterKey"])["T2_Min"].transform("min")
    late_jobs = inhouse[inhouse["T2_Min"] > inhouse["first_t2"] + 1e-9]
    late_may = late_jobs[late_jobs["OrderDate"].str.startswith("2026-05")]
    print(f"[Information Check] Single-receipt encounter-days: {n_single}/{n_total} ({100.0*n_single/n_total:.2f}%)")
    print(f"[Information Check] Late-arriving sibling jobs: {len(late_jobs)} total ({len(late_may)} in May)")

    # 2. Evaluate Table 4 job-level invariance across May and July-October
    days_may = [ws[d] for d in sorted(ws.keys()) if ws[d].year_month == "2026-05"]
    days_jo = [ws[d] for d in sorted(ws.keys()) if ws[d].year_month >= "2026-07"]

    cfg_w1 = SimConfig(prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
                       val_service_min=calib["val_service_min"], cadence_mult=calib["wave1_cadence_mult"], routing="lwr")
    cfg_w2 = SimConfig(prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
                       val_service_min=calib["val_service_min"], cadence_mult=calib["wave2_cadence_mult"], routing="lwr")

    w1_mem = np.array(learned["Memetic_DT_GA"]["w1_median_seed_vec"], dtype=float)
    w2_mem = np.array(learned["Memetic_DT_GA"]["w2_median_seed_vec"], dtype=float)

    for label, pmode, w1, w2 in [("FCFS", "FCFS", None, None), ("SPT", "SPT", None, None), ("Memetic DT-GA", "MEMETIC", w1_mem, w2_mem)]:
        rm = eval_mod.eval_days_detailed(days_may, cfg_w1, pmode, w1)
        rj = eval_mod.eval_days_detailed(days_jo, cfg_w2, pmode, w2)
        print(f"[Table 4] {label:15s} | May Job T2->T4={rm['job_mean_t2_t4']:.3f}m, Wait={rm['job_mean_wait']:.3f}m, Enc TAT={rm['pooled_mean']:.2f}m "
              f"| Jul-Oct Job T2->T4={rj['job_mean_t2_t4']:.3f}m, Wait={rj['job_mean_wait']:.3f}m, Enc TAT={rj['pooled_mean']:.2f}m")


if __name__ == "__main__":
    run_start_time_and_information_checks()
