"""
Step 1: Strict Chronological Twin Calibration (ZERO Test-Set Leakage).
Loads raw order-level extract `data/raw_10months_orders.csv.gz`:
- Wave 1 (Jan-Feb 2026 ONLY): estimates physical preparation lags alpha_g from quiet-hour floors,
  empirical T3->T4 service pools & TestType_ID median predictors hat_p, post-analytical residual
  reporting distribution, and calibrates per-section launch cadences tau_g strictly against
  Jan-Feb observed T2->T4 per resource group + overall FCFS TAT.
- Wave 2 (June 2026 ONLY): recalibrates post-April/June hardware and diurnal launch cadences
  strictly against June 2026 observations without touching May or July-October.
"""

import json
import os
import time
import zlib
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from des_engine import (
    GROUP_NAMES, GROUP_TO_ID, ANALYZER_SPECS,
    DayWorkload, SimConfig, simulate_day
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
RAW_ORDERS_GZ = os.path.join(DATA_DIR, "raw_10months_orders.csv.gz")
SYNTH_ORDERS_GZ = os.path.join(DATA_DIR, "synthetic_demo_orders.csv.gz")
CALIB_JSON = os.path.join(OUT_DIR, "calibrated_params.json")


def resolve_orders_path() -> str:
    """Resolve explicit DTGA_ORDERS_FILE, institutional extract if present locally, else open synthetic benchmark."""
    override = os.environ.get("DTGA_ORDERS_FILE", "").strip()
    if override and os.path.exists(override):
        return override
    if os.path.exists(RAW_ORDERS_GZ):
        return RAW_ORDERS_GZ
    if os.path.exists(SYNTH_ORDERS_GZ):
        return SYNTH_ORDERS_GZ
    raise FileNotFoundError(f"No order dataset found at {RAW_ORDERS_GZ} or {SYNTH_ORDERS_GZ}.")


def load_filtered_inhouse_orders():
    """Load orders (raw extract if available locally, otherwise open synthetic benchmark) and filter to in-house jobs."""
    orders_path = resolve_orders_path()
    if orders_path == SYNTH_ORDERS_GZ:
        print(f"[INFO] Restricted hospital extract absent; running on public synthetic benchmark: {orders_path}")
    df = pd.read_csv(orders_path)
    inhouse = df[(df["IsExternalOrder"] == 0) & (df["ResourceGroup"] != "Other")].copy()
    valid_mask = (
        inhouse["T2_Min"].notna()
        & inhouse["T4_Min"].notna()
        & (inhouse["T4_Min"] > inhouse["T2_Min"])
        & ((inhouse["T4_Min"] - inhouse["T2_Min"]) <= 360.0)
    )
    inhouse = inhouse[valid_mask].copy()
    inhouse["GroupID"] = inhouse["ResourceGroup"].map(GROUP_TO_ID).astype(int)
    return df, inhouse


def estimate_janfeb_distributions(cal_df: pd.DataFrame) -> dict:
    """Estimate quiet-floor prep lags alpha_g, test-type medians hat_p, and section T2->T4 targets from Jan-Feb ONLY."""
    prep_lags = {}
    t3t4_pools = {}
    pred_medians_by_type = {}
    pred_medians_by_group = {}
    target_t2_t4_mean = {}

    tmp = cal_df.copy()
    tmp["T2_T3"] = tmp["T3_Min"] - tmp["T2_Min"]
    tmp["T3_T4"] = tmp["T4_Min"] - tmp["T3_Min"]
    tmp["T2_T4"] = tmp["T4_Min"] - tmp["T2_Min"]
    tmp["Hour"] = (tmp["T2_Min"] // 60).astype(int)
    quiet = tmp[tmp["Hour"].isin([12, 14, 15])]

    for g_id, g_name in enumerate(GROUP_NAMES):
        sub = tmp[tmp["GroupID"] == g_id]
        qsub = quiet[quiet["GroupID"] == g_id]
        v23_q = qsub["T2_T3"][(qsub["T2_T3"] >= 1.0) & (qsub["T2_T3"] <= 120.0)].values
        v34 = sub["T3_T4"][(sub["T3_T4"] >= 1.5) & (sub["T3_T4"] <= 120.0)].values
        v24 = sub["T2_T4"][(sub["T2_T4"] >= 2.0) & (sub["T2_T4"] <= 240.0)].values
        v24_q = qsub["T2_T4"][(qsub["T2_T4"] >= 2.0) & (qsub["T2_T4"] <= 180.0)].values

        if len(v34) >= 50:
            # Physical serum reception + centrifugation floor measured directly at off-peak quiet hours (12:00/14:00/15:00)
            alpha_g = float(np.clip(np.median(v23_q) * 0.85, 8.0, 22.0)) if len(v23_q) >= 10 else 16.0
            pool_vals = np.round(v34, 2)
        else:
            # 1-way sections: infer active analytical duration from quiet-hour T2->T4 minus centrifugation/prep floor
            alpha_g = 15.0 if g_name == "Coagulation" else (12.0 if g_name == "Urinalysis" else 8.0)
            inferred_proc = np.clip(v24_q - alpha_g - 2.0 if len(v24_q) >= 20 else v24 - alpha_g - 5.0, 4.0, 90.0)
            pool_vals = np.round(inferred_proc, 2)

        prep_lags[str(g_id)] = round(alpha_g, 2)
        target_t2_t4_mean[str(g_id)] = round(float(np.mean(v24)), 2)

        rng = np.random.default_rng(100 + g_id)
        if len(pool_vals) > 2000:
            pool_vals = rng.choice(pool_vals, size=2000, replace=False)
        t3t4_pools[str(g_id)] = pool_vals.tolist()
        pred_medians_by_group[str(g_id)] = round(float(np.median(pool_vals)), 2)

    for tid, sub in tmp.groupby("TestType_ID"):
        g_id = int(sub["GroupID"].iloc[0])
        v34 = sub["T3_T4"][(sub["T3_T4"] >= 1.5) & (sub["T3_T4"] <= 120.0)].values
        if len(v34) >= 30:
            med = float(np.median(v34))
        else:
            v24 = sub["T2_T4"][(sub["T2_T4"] >= 2.0) & (sub["T2_T4"] <= 180.0)].values
            alpha_g = prep_lags[str(g_id)]
            med = float(np.median(np.clip(v24 - alpha_g - 4.0, 4.0, 90.0))) if len(v24) > 0 else pred_medians_by_group[str(g_id)]
        pred_medians_by_type[str(int(tid))] = round(med, 2)

    enc_cal = tmp.groupby(["OrderDate", "EncounterKey"]).agg(
        t4_max=("T4_Min", "max"),
        t5_max=("T5_Raw_Min", "max")
    ).reset_index()
    res_raw = enc_cal["t5_max"] - enc_cal["t4_max"]
    valid_res = np.clip(res_raw[enc_cal["t5_max"].notna()].values, 1.0, 180.0)
    rng = np.random.default_rng(999)
    res_sample = rng.choice(valid_res, size=min(2000, len(valid_res)), replace=False)

    return {
        "prep_lags": [prep_lags[str(i)] for i in range(5)],
        "target_janfeb_t2_t4_mean": target_t2_t4_mean,
        "pred_medians_by_group": pred_medians_by_group,
        "pred_medians_by_type": pred_medians_by_type,
        "t3t4_pools": t3t4_pools,
        "res_lag_sample": np.round(res_sample, 2).tolist(),
    }


def build_daily_workloads(inhouse: pd.DataFrame, calib_dict: dict, t4t5_cap: Optional[float] = 180.0) -> dict:
    """Convert filtered order rows into compact DayWorkload objects keyed by OrderDate."""
    pred_by_type = calib_dict["pred_medians_by_type"]
    pred_by_grp = calib_dict["pred_medians_by_group"]
    pools = {int(k): np.array(v, dtype=float) for k, v in calib_dict["t3t4_pools"].items()}
    res_pool = np.array(calib_dict["res_lag_sample"], dtype=float)
    q_val = calib_dict.get("val_service_min", 1.0)

    grp_arr = inhouse["GroupID"].values
    tid_arr = inhouse["TestType_ID"].values
    t3_arr = inhouse["T3_Min"].values
    t4_arr = inhouse["T4_Min"].values

    p_hat = np.array([
        pred_by_type.get(str(int(tid)), pred_by_grp[str(int(g))])
        for tid, g in zip(tid_arr, grp_arr)
    ], dtype=float)

    rng = np.random.default_rng(424242)
    p_tilde = np.empty(len(inhouse), dtype=float)
    diff34 = t4_arr - t3_arr
    valid_34 = np.isfinite(diff34) & (diff34 >= 1.5) & (diff34 <= 120.0)
    for i in range(len(inhouse)):
        if valid_34[i]:
            p_tilde[i] = float(diff34[i])
        else:
            pool = pools[int(grp_arr[i])]
            p_tilde[i] = float(pool[int(rng.integers(0, len(pool)))])

    inhouse_work = inhouse.copy()
    inhouse_work["p_hat"] = p_hat
    inhouse_work["p_tilde"] = p_tilde

    workloads = {}
    for odate, day_df in inhouse_work.groupby("OrderDate", sort=True):
        ym = str(day_df["YearMonth"].iloc[0])
        enc_codes, uniques = pd.factorize(day_df["EncounterKey"], sort=False)
        n_enc = len(uniques)
        n_jobs = len(day_df)

        j_t2 = day_df["T2_Min"].values.astype(float)
        j_t4 = day_df["T4_Min"].values.astype(float)
        j_t5 = day_df["T5_Raw_Min"].values.astype(float)

        enc_first_t2 = np.full(n_enc, 1e9, dtype=float)
        enc_max_t4 = np.full(n_enc, -1e9, dtype=float)
        enc_max_t5 = np.full(n_enc, np.nan, dtype=float)
        enc_counts = np.zeros(n_enc, dtype=np.int32)

        for idx in range(n_jobs):
            e = enc_codes[idx]
            enc_counts[e] += 1
            if j_t2[idx] < enc_first_t2[e]:
                enc_first_t2[e] = j_t2[idx]
            if j_t4[idx] > enc_max_t4[e]:
                enc_max_t4[e] = j_t4[idx]
            if np.isfinite(j_t5[idx]):
                if not np.isfinite(enc_max_t5[e]) or j_t5[idx] > enc_max_t5[e]:
                    enc_max_t5[e] = j_t5[idx]

        raw_res = enc_max_t5 - enc_max_t4
        valid_t5 = np.isfinite(enc_max_t5) & ((enc_max_t5 - enc_first_t2) >= 5.0) & ((enc_max_t5 - enc_first_t2) <= 600.0)

        enc_res_lag = np.empty(n_enc, dtype=float)
        obs_cap_list = []
        obs_uncap_list = []

        day_rng = np.random.default_rng(zlib.crc32(str(odate).encode("utf-8")))
        for e in range(n_enc):
            if np.isfinite(raw_res[e]):
                r_raw = max(0.0, float(raw_res[e]))
                r_c = min(r_raw, t4t5_cap) if t4t5_cap is not None else r_raw
                enc_res_lag[e] = max(0.0, r_c - q_val)
                if valid_t5[e]:
                    obs_uncap_list.append((enc_max_t4[e] - enc_first_t2[e]) + r_raw)
                    obs_cap_list.append((enc_max_t4[e] - enc_first_t2[e]) + r_c)
            else:
                r_draw = float(res_pool[int(day_rng.integers(0, len(res_pool)))])
                r_c = min(r_draw, t4t5_cap) if t4t5_cap is not None else r_draw
                enc_res_lag[e] = max(0.0, r_c - q_val)

        workloads[str(odate)] = DayWorkload(
            order_date=str(odate),
            year_month=ym,
            n_jobs=n_jobs,
            n_encounters=n_enc,
            job_enc=enc_codes.astype(np.int32),
            job_group=day_df["GroupID"].values.astype(np.int32),
            job_t2=j_t2,
            job_pred_dur=day_df["p_hat"].values.astype(float),
            job_real_dur=day_df["p_tilde"].values.astype(float),
            enc_first_t2=enc_first_t2,
            enc_job_counts=enc_counts,
            enc_res_lag=enc_res_lag,
            obs_tat_cap180=np.array(obs_cap_list, dtype=float),
            obs_tat_uncap=np.array(obs_uncap_list, dtype=float),
        )
    return workloads


def fit_section_cadences_janfeb(days_jf: List[DayWorkload], prep_lags: List[float], target_g_t2t4: dict) -> dict:
    """
    Independently calibrate effective morning/midday/afternoon launch cadence per resource section
    strictly on Jan-Feb 59 days so each section's queue buildup matches observed T2->T4.
    """
    cadence_epochs = {str(g): [1.0, 0.8, 0.7] for g in range(5)}
    ratios = [1.0, 0.78, 0.72]  # morning surge cadence -> midday/afternoon faster turnover

    for g_id in range(5):
        target_g = float(target_g_t2t4[str(g_id)])
        # Test scalar morning cadence c_m for section g_id independently (almost orthogonal across sections!)
        def err_g(base_c):
            cad_copy = {k: list(v) for k, v in cadence_epochs.items()}
            cad_copy[str(g_id)] = [base_c * r for r in ratios]
            cfg = SimConfig(prep_lags=prep_lags, cadence_epochs=cad_copy, val_service_min=1.0, routing="lwr")
            vals = [simulate_day(d, cfg, policy="FCFS").group_mean_t2_t4[g_id] for d in days_jf]
            return abs(float(np.mean(vals)) - target_g)

        hi_bound = 25.0 if g_id != 2 else 35.0
        res = minimize_scalar(err_g, bounds=(0.4, hi_bound), method="bounded", options={"maxiter": 14, "xatol": 0.05})
        best_c = float(res.x)
        cadence_epochs[str(g_id)] = [round(best_c * r, 3) for r in ratios]
        print(f"    -> Section {GROUP_NAMES[g_id]:<12} Target T2->T4={target_g:5.2f}m | Fitted morning cadence={best_c:5.2f}m")

    return cadence_epochs


def fit_global_cadence_scale(days_list: List[DayWorkload], base_cfg: SimConfig, target_mean_tat: float, routing: str, bounds=(0.4, 2.5)) -> dict:
    """Fit 1D global cadence multiplier inside calibration window to match Source Mean TAT."""
    cache = {}

    def eval_scale(sc: float) -> float:
        key = round(float(sc), 5)
        if key in cache:
            return cache[key][0]
        cfg = SimConfig(
            prep_lags=base_cfg.prep_lags, cadence_epochs=base_cfg.cadence_epochs,
            val_service_min=base_cfg.val_service_min, cadence_mult=float(sc), routing=routing,
        )
        tats = [simulate_day(d, cfg, policy="FCFS").tats for d in days_list]
        pooled = np.concatenate(tats)
        sim_mean = float(np.mean(pooled))
        sim_p90 = float(np.percentile(pooled, 90))
        sim_sla = float(np.mean(pooled > 120.0) * 100.0)
        err = abs(sim_mean - target_mean_tat)
        cache[key] = (err, sim_mean, sim_p90, sim_sla)
        return err

    res = minimize_scalar(eval_scale, bounds=bounds, method="bounded", options={"maxiter": 12, "xatol": 0.01})
    best_sc = float(res.x)
    _, sim_mean, sim_p90, sim_sla = cache[min(cache.keys(), key=lambda k: abs(k - best_sc))]
    return {
        "scale": round(best_sc, 4), "sim_mean": round(sim_mean, 2),
        "sim_p90": round(sim_p90, 2), "sim_sla": round(sim_sla, 2),
    }


def calibrate_all():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    _, inhouse = load_filtered_inhouse_orders()

    cal_jf = inhouse[inhouse["YearMonth"].isin(["2026-01", "2026-02"])].copy()
    stats_jf = estimate_janfeb_distributions(cal_jf)
    print(f"[WAVE 1] Prep lags alpha_g (from quiet hours): {stats_jf['prep_lags']}")

    stats_jf["cadence_epochs"] = {str(i): [1.0, 0.8, 0.7] for i in range(5)}
    stats_jf["val_service_min"] = 1.0

    jf_workloads = build_daily_workloads(cal_jf, stats_jf, t4t5_cap=180.0)
    jf_days = [jf_workloads[k] for k in sorted(jf_workloads.keys())]
    obs_jf = np.concatenate([d.obs_tat_cap180 for d in jf_days if len(d.obs_tat_cap180) > 0])
    target_jf_mean = float(np.mean(obs_jf))

    fitted_cadence = fit_section_cadences_janfeb(jf_days, stats_jf["prep_lags"], stats_jf["target_janfeb_t2_t4_mean"])
    stats_jf["cadence_epochs"] = fitted_cadence

    base_cfg_w1 = SimConfig(prep_lags=stats_jf["prep_lags"], cadence_epochs=fitted_cadence, routing="lwr")
    fit_w1 = fit_global_cadence_scale(jf_days, base_cfg_w1, target_jf_mean, routing="lwr", bounds=(0.5, 2.0))
    print(f"[WAVE 1 CALIBRATION - Jan-Feb 59d] Source Mean={target_jf_mean:.2f}m | Sim FCFS={fit_w1['sim_mean']:.2f}m (SLA={fit_w1['sim_sla']}%) | Scale={fit_w1['scale']}")

    # Wave 2 Recalibration on June 2026 ONLY (+/-15% bounded diurnal cadence adjustment around Wave 1 physics)
    cal_jun = inhouse[inhouse["YearMonth"] == "2026-06"].copy()
    jun_workloads = build_daily_workloads(cal_jun, stats_jf, t4t5_cap=180.0)
    jun_days = [jun_workloads[k] for k in sorted(jun_workloads.keys())]
    obs_jun = np.concatenate([d.obs_tat_cap180 for d in jun_days if len(d.obs_tat_cap180) > 0])
    target_jun_mean = float(np.mean(obs_jun))

    fit_w2 = fit_global_cadence_scale(jun_days, base_cfg_w1, target_jun_mean, routing="lwr", bounds=(0.82 * fit_w1["scale"], 1.15 * fit_w1["scale"]))
    print(f"[WAVE 2 RECALIBRATION - Jun 30d]    Source Mean={target_jun_mean:.2f}m | Sim FCFS={fit_w2['sim_mean']:.2f}m (SLA={fit_w2['sim_sla']}%) | Scale={fit_w2['scale']}")

    stats_jf["wave1_cadence_mult"] = fit_w1["scale"]
    stats_jf["wave1_fit_summary"] = fit_w1
    stats_jf["wave2_cadence_mult"] = fit_w2["scale"]
    stats_jf["wave2_fit_summary"] = fit_w2

    with open(CALIB_JSON, "w", encoding="utf-8") as f:
        json.dump(stats_jf, f, indent=2)
    print(f"[DONE] Saved physical per-section calibrated parameters in {time.time() - t0:.2f}s -> {CALIB_JSON}")


if __name__ == "__main__":
    calibrate_all()
