"""
Generate a purely synthetic (de novo non-clinical) benchmark dataset matching the exact
schema of `raw_10months_orders.csv.gz` so independent reviewers can verify the DES
and Memetic DT-GA code without proprietary hospital HIS-LIS access.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_SYNTH = os.path.join(BASE_DIR, "data", "synthetic_demo_orders.csv.gz")


def generate_synthetic_benchmark(seed: int = 202611):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", "2026-10-31", freq="D")
    groups = [
        ("Biochemistry", 1, 6, 18.0, 22.0),
        ("Hematology", 2, 19, 14.0, 12.0),
        ("Coagulation", 6, 7, 15.0, 19.0),
        ("Immunoassay", 4, 15, 16.0, 28.0),
        ("Urinalysis", 3, 22, 12.0, 14.0),
    ]
    rows = []
    tid_seq = 9000000
    for d in dates:
        odate = d.strftime("%Y-%m-%d")
        ym = d.strftime("%Y-%m")
        # Lighter synthetic benchmark (~23 encounters/day) for fast standalone code verification
        n_enc = 32 if ym == "2026-06" else 22
        for e in range(n_enc):
            enc_key = f"syn_{odate.replace('-','')}_{e:04d}"
            # Diurnal morning peak around 08:30 (510 min)
            t2_base = float(np.clip(rng.normal(540.0, 115.0), 420.0, 990.0))
            n_grp = int(rng.choice([1, 2, 3, 4], p=[0.28, 0.36, 0.24, 0.12]))
            chosen = rng.choice(len(groups), size=n_grp, replace=False)
            for gi in chosen:
                gname, ttype, dev_id, alpha, mu_dur = groups[int(gi)]
                tid_seq += 1
                t2 = t2_base + float(rng.uniform(0.0, 3.0))
                t3 = t2 + alpha + float(rng.gamma(2.0, 2.5)) if gname in ("Biochemistry", "Immunoassay") else np.nan
                proc = float(np.clip(rng.gamma(3.2, mu_dur / 3.2), 4.0, 85.0))
                t4 = (t3 if np.isfinite(t3) else (t2 + alpha)) + proc
                t5 = t4 + float(np.clip(rng.gamma(2.2, 14.0), 3.0, 160.0))
                rows.append({
                    "OrderDate": odate,
                    "YearMonth": ym,
                    "EncounterKey": enc_key,
                    "Test_ID": tid_seq,
                    "TestType_ID": ttype,
                    "Device_ID": dev_id,
                    "ParamCount": 1,
                    "IsExternalOrder": 0,
                    "SendOutDetailRows": 0,
                    "T2_Min": round(t2, 3),
                    "T3_Min": round(t3, 3) if np.isfinite(t3) else np.nan,
                    "T4_Min": round(t4, 3),
                    "T5_Raw_Min": round(t5, 3),
                    "ResourceGroup": gname,
                })
    df = pd.DataFrame(rows)
    # Add tracking offset columns using a separate PRNG stream so core 15,349 rows stay identical
    rng_rec = np.random.default_rng(seed + 1)
    df["Test_minus_Require_Min"] = 0.0
    df["Test_minus_Collected_Min"] = 0.0
    df["Device_minus_Test_Min"] = np.round(df["T3_Min"] - df["T2_Min"], 3)
    rec_raw = rng_rec.gamma(3.0, 6.5, size=len(df))
    has_scan = df["OrderDate"] >= "2026-07-21"
    rec_diff = np.where(has_scan, np.round(np.clip(rec_raw, 2.0, np.maximum(3.0, df["T4_Min"] - df["T2_Min"] - 1.0)), 3), np.nan)
    df["Received_minus_Test_Min"] = rec_diff
    df["Received_Min_Of_Day"] = np.round(df["T2_Min"] + rec_diff, 3)

    os.makedirs(os.path.dirname(OUT_SYNTH), exist_ok=True)
    df.to_csv(OUT_SYNTH, index=False, compression="gzip", encoding="utf-8")
    print(f"[DONE] Generated purely synthetic 304-day de novo benchmark: {len(df):,} rows -> {OUT_SYNTH} ({os.path.getsize(OUT_SYNTH):,} bytes)")


if __name__ == "__main__":
    generate_synthetic_benchmark()
