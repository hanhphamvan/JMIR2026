# Memetic Digital Twin–Genetic Algorithm (DT-GA) for Outpatient Laboratory Turnaround Time Simulation

Public reproducibility repository accompanying the JMIR manuscript evaluating a **two-wave Fork–Join discrete-event simulation (DES) and Memetic Digital Twin–Genetic Algorithm (DT-GA)** dispatch framework for outpatient clinical laboratory turnaround time (TAT) and >120-minute SLA compliance.

---

## 1. Repository Contents & Zero-PHI Data Policy

This repository contains **only** the complete simulation/optimization source code and a **purely synthetic, non-clinical 304-day schema-matched benchmark dataset** packaged without precomputed outputs:

```text
├── algorithm/
│   ├── 01_extract_raw_from_sql.py       # Hardened SQL Server extractor (env-var credentials + keyed HMAC-SHA256)
│   ├── 02_calibrate_twin_parameters.py  # Step 1: Chronological Wave 1 (Jan–Feb) & Wave 2 (Jun) calibration
│   ├── 03_optimize_policies.py          # Step 2: Equal-budget metaheuristic training (112 evals/seed × 6 seeds)
│   ├── 04_evaluate_heldout_and_stress.py# Step 3: Frozen 154-day held-out DES, ablations, sensitivity, stress & audits
│   ├── 05_statistical_benchmark_report.py# Step 4: 7-day moving-block bootstrap CIs, Wilcoxon-Holm tests & report
│   ├── des_engine.py                    # Core heapq Fork–Join DES engine (14 analyzer modules across 5 sections, LWR vs fixed-scan, Kleinrock T2→T4 tracking)
│   ├── generate_synthetic_benchmark.py  # Standalone de novo synthetic 304-day generator (PRNG seed 202611)
│   └── run_all_from_scratch.py          # Single-command end-to-end execution script (PYTHONHASHSEED=0)
├── data/
│   └── synthetic_demo_orders.csv.gz     # De novo synthetic 304-day schema-matched dataset (15,349 synthetic rows)
├── outputs/
│   └── .gitkeep                         # Clean output directory populated only when executed
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

### Data Governance Statement
- **Included publicly (`data/synthetic_demo_orders.csv.gz`)**: Mathematically generated de novo by `algorithm/generate_synthetic_benchmark.py` (NumPy PCG64 PRNG seed `202611`; 15,349 synthetic rows across Jan 1 – Oct 31, 2026) without copying, transforming, or linking any real patient rows.
- **Excluded from public release**: The retrospective institutional HIS-LIS extract (`raw_10months_orders.csv.gz`; 3,563,568 clinical rows) and hospital send-out test catalogue (`danhmucguimau.xlsx`; 241 codes) contain institutional operational and pseudonymized patient-linked records and are not publicly posted. Qualified researchers seeking institutional data access may contact **Pham Van Hanh** (`hanhpv109@hlu.edu.vn`), subject to written authorization from Hanoi Andrology and Fertility Hospital.

---

## 2. Tested Environment & Dependencies

Manuscript analyses were executed with:
- **Python**: `3.13.3`
- **NumPy**: `2.2.4`
- **pandas**: `2.2.3`
- **SciPy**: `1.17.1`
- **openpyxl**: `3.1.5`
- **pyodbc** (optional, required only for `01_extract_raw_from_sql.py` on an authorized hospital SQL Server)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. Reproducing the Pipeline on the Open Synthetic Benchmark

Running `run_all_from_scratch.py` executes Steps 1–4 chronologically from scratch (`PYTHONHASHSEED=0`, deterministic `zlib.crc32` encounter seeding). When institutional `data/raw_10months_orders.csv.gz` is absent, Step 1 automatically falls back to `data/synthetic_demo_orders.csv.gz`:

```bash
python algorithm/run_all_from_scratch.py
```

To regenerate the exact included synthetic benchmark (`15,349` rows, PRNG seed `202611`):
```bash
python algorithm/generate_synthetic_benchmark.py
```

Generated artifacts written to `outputs/`:
- `outputs/calibrated_params.json` — Wave 1 & Wave 2 calibrated preparation lags, section cadences, and empirical duration pools
- `outputs/learned_weights.json` — 6-seed learned chromosomes and April/June-selected vectors across metaheuristics & ablations
- `outputs/daily_simulation_results.csv` — Day-by-day paired held-out outcomes across all comparators (May + July–October)
- `outputs/full_benchmark_outcomes.json` — Pooled metrics, ablations, weight/cap sensitivity, 30-replicate resampling, stress tests, analyzer utilization, and Kleinrock conservation verification
- `outputs/COMPARISON_VS_MANUSCRIPT.md` — Summary report with paired 7-day moving-block bootstrap CIs and Holm-adjusted Wilcoxon tests

---

## 4. Institutional Extraction Script (Authorized Local Use Only)

`algorithm/01_extract_raw_from_sql.py` documents the exact SQL extraction schema and cohort filtering rules. For future institutional runs, it requires environment-variable credentials and a secret HMAC key before execution:

```bash
export HIS_SQL_SERVER="<sql-server-host>"
export HIS_SQL_UID="<read-only-user>"
export HIS_SQL_PWD="<password>"
export HIS_HMAC_KEY="<secret-hmac-key>"
python algorithm/01_extract_raw_from_sql.py
```

---

## 5. License

Released under the [MIT License](LICENSE).
