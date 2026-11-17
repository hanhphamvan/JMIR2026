"""
Diagnostic Analysis 4: Physical Laboratory Scan-In (RECEIVED in DT_SAMPLE_TRACKING_EVENTS),
May 11 Batch-Finalization Exclusion, Empirical T3–T4 Peak Onboard Occupancy, and
Tight Onboard Capacity (c_m <= 10 and c_m <= 6) Where Analyzer Start Times Depend on Dispatch Policy.
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
eval_mod = SourceFileLoader("eval_mod", os.path.join(ALG_DIR, "04_evaluate_heldout_and_stress.py")).load_module()


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
    ws = calib_mod.build_daily_workloads(inhouse, calib, t4t5_cap=180.0)

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
        print(f"[Tight Onboard c_m <= {cap_lim}] FCFS Job T2->T4={rf['job_mean_t2_t4']}m (Enc {rf['pooled_mean']}m, SLA {rf['pooled_sla_pct']}%) | "
              f"SPT Job T2->T4={rs['job_mean_t2_t4']}m (Enc {rs['pooled_mean']}m, p90 {rs['pooled_p90']}m, SLA {rs['pooled_sla_pct']}%) | "
              f"Memetic Job T2->T4={rm['job_mean_t2_t4']}m (Enc {rm['pooled_mean']}m, SLA {rm['pooled_sla_pct']}%)")
    for i, s in enumerate(orig_specs):
        des.ANALYZER_SPECS[i]["cap"] = s["cap"]


if __name__ == "__main__":
    audit_empirical_onboard_and_tight_capacity()
