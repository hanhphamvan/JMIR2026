# Does Specimen Dispatch Optimization Shorten Outpatient Laboratory Turnaround Time? A 10-Month Two-Wave Fork–Join Simulation Study

Public reproducibility repository accompanying the JMIR manuscript evaluating a **two-wave Fork–Join discrete-event simulation (DES) and Memetic Digital Twin–Genetic Algorithm (Memetic DT-GA)** dispatch framework for outpatient clinical laboratory turnaround time (TAT) and >120-minute service-level agreement (SLA) compliance.

---

## 1. Repository Layout & Zero-PHI Data Policy

This repository contains the complete simulation and optimization source code, standalone diagnostic analysis scripts, and a **purely synthetic, non-clinical 304-day schema-matched benchmark dataset** (`data/synthetic_demo_orders.csv.gz`; 15,349 rows generated with PRNG seed `202611`, containing zero patient records):

```text
├── algorithm/
│   ├── 01_extract_raw_from_sql.py         # SQL Server extractor (env-var credentials + keyed HMAC-SHA256)
│   ├── 02_calibrate_twin_parameters.py    # Wave 1 (Jan–Feb) & Wave 2 (June) parameter calibration
│   ├── 03_optimize_policies.py            # 6-seed optimization (Memetic DT-GA, No-Var, DE, RS, Std Linear, ablations)
│   ├── 04_evaluate_heldout_and_stress.py  # Held-out replay (May 31d + Jul–Oct 123d), ablations, capacity scenarios
│   ├── 05_statistical_benchmark_report.py # Paired daily contrasts, 7-day circular MBB 95% CIs, Holm adjustment
│   ├── des_engine.py                      # Core Fork–Join DES engine & deterministic dispatch decoder
│   ├── generate_synthetic_benchmark.py    # De novo synthetic 304-day dataset generator (seed 202611)
│   └── run_all_from_scratch.py            # End-to-end pipeline runner
├── analysis_scripts/
│   ├── 01_audit_start_time_invariance.py  # Start-time invariance (Eq. 13) & information checks
│   ├── 02_audit_wilcoxon_permutation.py   # Sign-flip permutation Wilcoxon tests & 2-level bootstrap CIs
│   ├── 03_audit_calibration_lag_sensitivity.py # Analytic T2–T4 vs T2–T5 replay vs calibration-period lag draws
│   ├── 04_audit_received_scan_and_tight_capacity.py # Empirical T3–T4 onboard peaks, May 11 exclusion, tight capacity (c_m<=10,6) & re-optimization
│   └── README.md
├── data/
│   └── synthetic_demo_orders.csv.gz       # Synthetic 304-day benchmark dataset (15,349 rows; zero PHI)
├── outputs/
│   └── .gitkeep
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

---

## 2. Environment & Execution

- **Python**: `3.13.3` (`PYTHONHASHSEED=0`)
- **Packages**: `numpy==2.2.4`, `pandas==2.2.3`, `scipy==1.17.1`, `openpyxl==3.1.5`

```bash
pip install -r requirements.txt
export PYTHONHASHSEED=0
python algorithm/run_all_from_scratch.py
```
