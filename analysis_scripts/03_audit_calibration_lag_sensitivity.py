"""
Diagnostic Analysis 3: Simulator Fidelity Separately for Analytic Completion (T2–T4),
Conditional Replay (T2–T5), and Calibration-Period Post-Analytic Delays (Table 3),
and 11-Policy Evaluation Under Calibration-Period Delays (Table 7).
"""

import os
import sys
import json
import zlib
import numpy as np
import pandas as pd
from scipy import stats

ALG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "algorithm")
sys.path.insert(0, ALG_DIR)

from des_engine import SimConfig, DayWorkload
from importlib.machinery import SourceFileLoader

calib_mod = SourceFileLoader("calib_mod", os.path.join(ALG_DIR, "02_calibrate_twin_parameters.py")).load_module()
eval_mod = SourceFileLoader("eval_mod", os.path.join(ALG_DIR, "04_evaluate_heldout_and_stress.py")).load_module()


def make_analytic_only_days(days, q_val=1.0):
    """Set residual lag r_i = -q_val so that F_i = v_i + q_val + r_i = v_i (analytic completion T2->T4)."""
    return [
        DayWorkload(
            d.order_date, d.year_month, d.n_jobs, d.n_encounters,
            d.job_enc, d.job_group, d.job_t2, d.job_pred_dur, d.job_real_dur,
            d.enc_first_t2, d.enc_job_counts,
            np.full(d.n_encounters, -q_val, dtype=float),
            d.obs_tat_cap180, d.obs_tat_uncap
        )
        for d in days
    ]


def make_calibration_lag_days(days, cal_pool: np.ndarray, q_val: float = 1.0):
    """Draw post-analytic delay from calibration-period pool using date-seeded PCG64 generator shared across policies."""
    out = []
    for d in days:
        rng = np.random.default_rng(zlib.crc32(str(d.order_date).encode("utf-8")))
        draws = cal_pool[rng.integers(0, len(cal_pool), size=d.n_encounters)]
        res_lag = np.maximum(0.0, draws - q_val)
        out.append(
            DayWorkload(
                d.order_date, d.year_month, d.n_jobs, d.n_encounters,
                d.job_enc, d.job_group, d.job_t2, d.job_pred_dur, d.job_real_dur,
                d.enc_first_t2, d.enc_job_counts,
                res_lag, d.obs_tat_cap180, d.obs_tat_uncap
            )
        )
    return out


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calib_path = os.path.join(base_dir, "outputs", "calibrated_params.json")
    weights_path = os.path.join(base_dir, "outputs", "learned_weights.json")
    if os.path.exists(calib_path) and os.path.exists(weights_path):
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        with open(weights_path, "r", encoding="utf-8") as f:
            learned = json.load(f)
        _, inhouse = calib_mod.load_filtered_inhouse_orders()
        ws = calib_mod.build_daily_workloads(inhouse, calib, t4t5_cap=180.0)
        days_may = [ws[d] for d in sorted(ws.keys()) if ws[d].year_month == "2026-05"]
        cfg_w1 = SimConfig(prep_lags=calib["prep_lags"], cadence_epochs=calib["cadence_epochs"],
                           val_service_min=calib["val_service_min"], cadence_mult=calib["wave1_cadence_mult"], routing="lwr")
        r_t24 = eval_mod.eval_days_detailed(make_analytic_only_days(days_may, calib["val_service_min"]), cfg_w1, "FCFS")
        print(f"May T2->T4 Simulated FCFS: mean={r_t24['pooled_mean']}m, p90={r_t24['pooled_p90']}m, >120m={r_t24['pooled_sla_pct']}%")
    else:
        print("Calibration-period post-analytic delay and T2->T4 fidelity module ready.")
