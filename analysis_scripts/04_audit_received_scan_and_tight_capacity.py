"""
Diagnostic Analysis 4:
  1. T2 Cross-Check Against T_REG_LIST (Tube_Collected_Date, DeviceReceived_Date) and Physical
     Laboratory Scan-In (RECEIVED in DT_SAMPLE_TRACKING_EVENTS), plus Wave 2 Scanned-Receipt Simulation.
  2. Empirical T3–T4 Peak Concurrent Tubes on Board (Filtered <=120 min and Unfiltered T3 < T4)
     Across the 6 Two-Way Biochemistry and Immunoassay Analyzers.
  3. May 11 Batch-Finalization Exclusion Sensitivity.
  4. Tight Onboard & Coagulation Batch Capacity (c_m <= 10 and c_m <= 6), Equation 10 Objective
     Evaluation on the 8 Training Days / All March–April Days / May Test Days, and 6-Seed
     Memetic DT-GA Re-Optimization Under c_m <= 10 (SEEDS = [101, 202, 303, 404, 505, 606]).
"""

import os
import sys
import copy
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALG_DIR = os.path.join(BASE_DIR, "algorithm")
sys.path.insert(0, ALG_DIR)

import des_engine as des
from des_engine import SimConfig
from importlib.machinery import SourceFileLoader

calib_mod = SourceFileLoader("calib_mod", os.path.join(ALG_DIR, "02_calibrate_twin_parameters.py")).load_module()
opt_mod = SourceFileLoader("opt_mod", os.path.join(ALG_DIR, "03_optimize_policies.py")).load_module()
eval_mod = SourceFileLoader("eval_mod", os.path.join(ALG_DIR, "04_evaluate_heldout_and_stress.py")).load_module()

DEVICE_NAMES = {
    6: "Advia1650",
    14: "Cobas_D1",
    15: "E601",
    17: "E601_1",
    18: "AU680",
    21: "C501_E601",
}


def ensure_tracking_columns(inhouse: pd.DataFrame) -> pd.DataFrame:
    """Ensure tracking timestamp offset columns exist on inhouse DataFrame."""
    track_cols = [
        "Test_minus_Require_Min",
        "Test_minus_Collected_Min",
        "Device_minus_Test_Min",
        "Received_minus_Test_Min",
        "Received_Min_Of_Day",
    ]
    if all(c in inhouse.columns for c in track_cols):
        return inhouse

    track_gz = os.path.join(BASE_DIR, "data", "test_id_tracking_timestamps.csv.gz")
    if os.path.exists(track_gz):
        tdf = pd.read_csv(track_gz).drop_duplicates("Test_ID")
        avail = [c for c in track_cols if c in tdf.columns]
        return inhouse.merge(tdf[["Test_ID"] + avail], on="Test_ID", how="left")

    return inhouse


def audit_t2_and_received_scan(inhouse: pd.DataFrame, calib: dict, learned: dict):
    """Part 1: Audit T2 against Tube_Collected_Date / Require_Date / DeviceReceived_Date and run
    the Wave 2 physical laboratory barcode scan-in (RECEIVED) sensitivity analysis."""
    df = ensure_tracking_columns(inhouse)
    if "Received_Min_Of_Day" not in df.columns:
        print("[Part 1] Tracking columns absent in dataset; skipping RECEIVED scan-in sensitivity.")
        return

    for col, label in [
        ("Test_minus_Require_Min", "T.Test_Date - T.Require_Date"),
        ("Test_minus_Collected_Min", "T.Test_Date - Tube_Collected_Date"),
        ("Device_minus_Test_Min", "DeviceReceived_Date (T3) - T.Test_Date"),
    ]:
        if col in df.columns:
            s = df[col].dropna()
            s = s[(s >= -30.0) & (s <= 240.0)]
            if len(s) > 0:
                print(f"[T2 Audit] {label:38s}: n={len(s):6d} | median={np.median(s):.1f}m "
                      f"(IQR {np.percentile(s, 25):.1f} to {np.percentile(s, 75):.1f}m)")

    jo = df[df["YearMonth"] >= "2026-07"].copy()
    any_rec = jo["Received_Min_Of_Day"].notna()
    s_rec_240 = jo["Received_minus_Test_Min"].dropna()
    s_rec_240 = s_rec_240[(s_rec_240 >= -30.0) & (s_rec_240 <= 240.0)]
    valid_rec = (
        jo["Received_Min_Of_Day"].notna()
        & (jo["Received_minus_Test_Min"] >= 0.0)
        & (jo["Received_Min_Of_Day"] < jo["T4_Min"])
    )
    print(f"[RECEIVED Audit] Jul-Oct total jobs: {len(jo):,} | matched to same-day RECEIVED: {int(any_rec.sum()):,} "
          f"across {jo.loc[any_rec, 'OrderDate'].nunique()} active logging days")
    if len(s_rec_240) > 0:
        print(f"[RECEIVED Audit] Jobs with -30 <= T_rec - T2 <= 240m: {len(s_rec_240):,} | "
              f"median={np.median(s_rec_240):.1f}m (IQR {np.percentile(s_rec_240, 25):.1f} to {np.percentile(s_rec_240, 75):.1f}m)")

    jo_rec = jo[valid_rec].copy()
    jo_rec["T2_Min"] = jo_rec["Received_Min_Of_Day"]
    ws_rec = calib_mod.build_daily_workloads(jo_rec, calib, t4t5_cap=180.0)
    days_rec = [ws_rec[d] for d in sorted(ws_rec.keys())]
    n_enc_rec = sum(d.n_encounters for d in days_rec)
    n_jobs_rec = sum(d.n_jobs for d in days_rec)
    print(f"[RECEIVED Sim Cohort] Causal T2 <= T_rec < T4: {n_jobs_rec:,} jobs across {n_enc_rec:,} encounter-days on {len(days_rec)} days")

    if not days_rec:
        return

    obs_all = np.concatenate([d.obs_tat_cap180 for d in days_rec if len(d.obs_tat_cap180) > 0])
    print(f"[RECEIVED Observed TAT] Mean={np.mean(obs_all):.2f}m | p90={np.percentile(obs_all, 90):.2f}m | "
          f">120m={100.0 * np.mean(obs_all > 120.0):.2f}%")

    w1_mem = np.array(learned["Memetic_DT_GA"]["w1_median_seed_vec"], dtype=float)
    w2_mem = np.array(learned["Memetic_DT_GA"]["w2_median_seed_vec"], dtype=float)

    for mode_lbl, p_lags in [
        ("max(2.0, alpha_g - 12.0)", [max(2.0, x - 12.0) for x in calib["prep_lags"]]),
        ("Unadjusted alpha_g", calib["prep_lags"]),
    ]:
        cfg = SimConfig(
            prep_lags=p_lags,
            cadence_epochs=calib["cadence_epochs"],
            val_service_min=calib["val_service_min"],
            cadence_mult=calib["wave2_cadence_mult"],
            routing="lwr",
        )
        r_fcfs = eval_mod.eval_days_detailed(days_rec, cfg, "FCFS")
        r_w2 = eval_mod.eval_days_detailed(days_rec, cfg, "MEMETIC", w2_mem)
        r_w1 = eval_mod.eval_days_detailed(days_rec, cfg, "MEMETIC", w1_mem)
        print(f"[RECEIVED Wave 2 Sim - {mode_lbl}] "
              f"FCFS: Job T_rec->T4={r_fcfs['job_mean_t2_t4']:.3f}m, Wait={r_fcfs['job_mean_wait']:.3f}m, "
              f"Enc Mean={r_fcfs['pooled_mean']:.2f}m, p90={r_fcfs['pooled_p90']:.2f}m, SLA={r_fcfs['pooled_sla_pct']:.2f}% ({r_fcfs['pooled_sla_cnt']}) | "
              f"w*W2: Enc Mean={r_w2['pooled_mean']:.2f}m, p90={r_w2['pooled_p90']:.2f}m, SLA={r_w2['pooled_sla_pct']:.2f}% ({r_w2['pooled_sla_cnt']}) | "
              f"w*W1: Enc Mean={r_w1['pooled_mean']:.2f}m, SLA={r_w1['pooled_sla_pct']:.2f}% ({r_w1['pooled_sla_cnt']})")


def audit_empirical_onboard_peaks(inhouse: pd.DataFrame):
    """Part 2: Compute filtered (0 < T4_Min - T3_Min <= 120m; n=155,327) and unfiltered (T3_Min < T4_Min; n=157,758)
    daily peak simultaneous tubes on board across the 6 two-way biochemistry and immunoassay analyzers."""
    valid = inhouse[
        inhouse["T3_Min"].notna()
        & inhouse["T4_Min"].notna()
        & (inhouse["T4_Min"] > inhouse["T3_Min"])
    ].copy()
    valid["dur_m"] = valid["T4_Min"] - valid["T3_Min"]

    for label, sub in [
        ("Filtered (0 < T4-T3 <= 120m)", valid[valid["dur_m"] <= 120.0]),
        ("Unfiltered (all T3 < T4)", valid),
    ]:
        summary = {}
        for dev_id, grp in sub.groupby("Device_ID"):
            if int(dev_id) not in DEVICE_NAMES:
                continue
            daily_peaks = []
            for _, dsub in grp.groupby("OrderDate"):
                ev_t = np.concatenate([dsub["T3_Min"].values, dsub["T4_Min"].values])
                ev_d = np.concatenate([np.ones(len(dsub), dtype=int), -np.ones(len(dsub), dtype=int)])
                order = np.lexsort((ev_d, ev_t))
                daily_peaks.append(int(np.max(np.cumsum(ev_d[order]))))
            pk_arr = np.array(daily_peaks)
            summary[DEVICE_NAMES[int(dev_id)]] = {
                "jobs": len(grp),
                "median": int(np.median(pk_arr)),
                "p90": int(np.percentile(pk_arr, 90)),
                "max": int(np.max(pk_arr)),
            }
        print(f"[Empirical T3-T4 Peak Onboard - {label}] total_jobs={len(sub):,}: {summary}")


def audit_may11_and_tight_capacity(inhouse: pd.DataFrame, calib: dict, learned: dict):
    """Part 3 & 4: May 11 exclusion, tight capacity (c_m <= 10 and c_m <= 6), Eq. 10 objective
    losses on 8 training days / all March–April days / May days, and 6-seed re-optimization."""
    ws_all = calib_mod.build_daily_workloads(inhouse, calib, t4t5_cap=180.0)
    # Use the exact training subset construction as in 03_optimize_policies.py (lines 269-274)
    train_subset = inhouse[inhouse["YearMonth"].isin(["2026-03", "2026-04", "2026-06"])].copy()
    ws_train = calib_mod.build_daily_workloads(train_subset, calib, t4t5_cap=180.0)
    days_mar = opt_mod.stratified_days(ws_train, ["2026-03"], n_pick=4)
    days_apr = opt_mod.stratified_days(ws_train, ["2026-04"], n_pick=4)
    days_jun = opt_mod.stratified_days(ws_train, ["2026-06"], n_pick=4)
    days_train_8 = days_mar + days_apr
    days_mar_apr_all = [ws_all[d] for d in sorted(ws_all.keys()) if ws_all[d].year_month in ("2026-03", "2026-04")]
    days_may = [ws_all[d] for d in sorted(ws_all.keys()) if ws_all[d].year_month == "2026-05"]
    days_may_no11 = [ws_all[d] for d in sorted(ws_all.keys()) if ws_all[d].year_month == "2026-05" and d != "2026-05-11"]

    cfg_w1 = SimConfig(
        prep_lags=calib["prep_lags"],
        cadence_epochs=calib["cadence_epochs"],
        val_service_min=calib["val_service_min"],
        cadence_mult=calib["wave1_cadence_mult"],
        routing="lwr",
    )
    cfg_w2 = SimConfig(
        prep_lags=calib["prep_lags"],
        cadence_epochs=calib["cadence_epochs"],
        val_service_min=calib["val_service_min"],
        cadence_mult=calib["wave2_cadence_mult"],
        routing="lwr",
    )
    w1_mem = np.array(learned["Memetic_DT_GA"]["w1_median_seed_vec"], dtype=float)

    rf_no11 = eval_mod.eval_days_detailed(days_may_no11, cfg_w1, "FCFS")
    rm_no11 = eval_mod.eval_days_detailed(days_may_no11, cfg_w1, "MEMETIC", w1_mem)
    print(f"[May excl. May 11 (30d)] FCFS: {rf_no11['pooled_mean']:.2f}m / {rf_no11['pooled_sla_pct']:.2f}% ({rf_no11['pooled_sla_cnt']}) "
          f"| Memetic: {rm_no11['pooled_mean']:.2f}m / {rm_no11['pooled_sla_pct']:.2f}% ({rm_no11['pooled_sla_cnt']})")

    orig_specs = copy.deepcopy(des.ANALYZER_SPECS)

    # Sequential-only c_m <= 10 check (leaving coagulation batch size at 25)
    for i, s in enumerate(des.ANALYZER_SPECS):
        if s["mode"] == "seq":
            s["cap"] = min(orig_specs[i]["cap"], 10)
        else:
            s["cap"] = orig_specs[i]["cap"]
    rf_seq10 = eval_mod.eval_days_detailed(days_may, cfg_w1, "FCFS")
    print(f"[Sequential-Only c_m <= 10 (Coag batch=25)] FCFS May: Mean={rf_seq10['pooled_mean']:.2f}m, SLA={rf_seq10['pooled_sla_pct']:.2f}% ({rf_seq10['pooled_sla_cnt']})")

    # Combined sequential onboard and coagulation batch cap at c_m <= 10 and c_m <= 6
    for cap_lim in (10, 6):
        for i, s in enumerate(des.ANALYZER_SPECS):
            s["cap"] = min(orig_specs[i]["cap"], cap_lim)
        rf = eval_mod.eval_days_detailed(days_may, cfg_w1, "FCFS")
        rs = eval_mod.eval_days_detailed(days_may, cfg_w1, "SPT")
        rm = eval_mod.eval_days_detailed(days_may, cfg_w1, "MEMETIC", w1_mem)
        print(f"[Tight Onboard & Batch c_m <= {cap_lim}] "
              f"FCFS Job T2->T4={rf['job_mean_t2_t4']:.3f}m (Wait {rf['job_mean_wait']:.3f}m, Enc {rf['pooled_mean']:.2f}m, p90 {rf['pooled_p90']:.2f}m, SLA {rf['pooled_sla_pct']:.2f}% [{rf['pooled_sla_cnt']}]) | "
              f"SPT Job T2->T4={rs['job_mean_t2_t4']:.3f}m (Wait {rs['job_mean_wait']:.3f}m, Enc {rs['pooled_mean']:.2f}m, p90 {rs['pooled_p90']:.2f}m, SLA {rs['pooled_sla_pct']:.2f}% [{rs['pooled_sla_cnt']}]) | "
              f"Memetic (frozen) Job T2->T4={rm['job_mean_t2_t4']:.3f}m (Wait {rm['job_mean_wait']:.3f}m, Enc {rm['pooled_mean']:.2f}m, p90 {rm['pooled_p90']:.2f}m, SLA {rm['pooled_sla_pct']:.2f}% [{rm['pooled_sla_cnt']}])")

        if cap_lim == 10:
            loss_fcfs_8 = opt_mod.compute_loss(None, days_train_8, cfg_w1, policy="FCFS")
            loss_spt_8 = opt_mod.compute_loss(None, days_train_8, cfg_w1, policy="SPT")
            loss_fcfs_ma = opt_mod.compute_loss(None, days_mar_apr_all, cfg_w1, policy="FCFS")
            loss_spt_ma = opt_mod.compute_loss(None, days_mar_apr_all, cfg_w1, policy="SPT")
            loss_fcfs_may = opt_mod.compute_loss(None, days_may, cfg_w1, policy="FCFS")
            loss_spt_may = opt_mod.compute_loss(None, days_may, cfg_w1, policy="SPT")
            print(f"[Eq. 10 Objective under c_m <= 10] "
                  f"8 Training Days: SPT={loss_spt_8:.1f} vs FCFS={loss_fcfs_8:.1f} | "
                  f"All Mar-Apr Days: SPT={loss_spt_ma:.1f} vs FCFS={loss_fcfs_ma:.1f} | "
                  f"May Test Days: SPT={loss_spt_may:.1f} vs FCFS={loss_fcfs_may:.1f}")

            seed_rows = []
            for seed in opt_mod.SEEDS:
                _, _, w_w1, f_w1, _, _ = opt_mod._run_single_seed_task((
                    "Memetic_c10", seed, days_mar, days_apr, days_jun,
                    cfg_w1, cfg_w2, "MEMETIC", 0.35, 90.0, 0.25, True, True, True, 10, "ga"
                ))
                rm_reopt = eval_mod.eval_days_detailed(days_may, cfg_w1, "MEMETIC", np.array(w_w1, dtype=float))
                seed_rows.append((seed, round(f_w1, 3), rm_reopt["pooled_mean"], rm_reopt["pooled_p90"], rm_reopt["pooled_sla_pct"], rm_reopt["pooled_sla_cnt"]))
            med_idx = int(np.argsort([r[1] for r in seed_rows])[len(seed_rows) // 2])
            print(f"[Re-optimized Memetic DT-GA under c_m <= 10 across 6 seeds {opt_mod.SEEDS}]:")
            for r in seed_rows:
                print(f"  seed={r[0]} | Apr_loss={r[1]:.3f} | May Mean={r[2]:.2f}m, p90={r[3]:.2f}m, SLA={r[4]:.2f}% ({r[5]})")
            print(f"  -> Median-loss seed: {seed_rows[med_idx]} | SLA range across seeds: "
                  f"{min(r[4] for r in seed_rows):.2f}% to {max(r[4] for r in seed_rows):.2f}%")

    for i, s in enumerate(orig_specs):
        des.ANALYZER_SPECS[i]["cap"] = s["cap"]


def main():
    out_dir = os.path.join(BASE_DIR, "outputs")
    alt_out_dir = os.path.join(os.path.dirname(BASE_DIR), "verified_10months_des_outputs")
    calib_path = os.path.join(out_dir, "calibrated_params.json")
    weights_path = os.path.join(out_dir, "learned_weights.json")
    if not (os.path.exists(calib_path) and os.path.exists(weights_path)):
        calib_path = os.path.join(alt_out_dir, "calibrated_params.json")
        weights_path = os.path.join(alt_out_dir, "learned_weights.json")
    if not (os.path.exists(calib_path) and os.path.exists(weights_path)):
        print("Calibration and learned weight files required in outputs/.")
        return

    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    with open(weights_path, "r", encoding="utf-8") as f:
        learned = json.load(f)

    _, inhouse = calib_mod.load_filtered_inhouse_orders()
    audit_t2_and_received_scan(inhouse, calib, learned)
    audit_empirical_onboard_peaks(inhouse)
    audit_may11_and_tight_capacity(inhouse, calib, learned)


if __name__ == "__main__":
    main()
