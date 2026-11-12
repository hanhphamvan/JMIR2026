"""
One-Click End-to-End From-Scratch Execution Runner.
Runs:
  Step 1: 02_calibrate_twin_parameters.py (Strict Jan-Feb & June calibration)
  Step 2: 03_optimize_policies.py (From-scratch 112-eval optimization across 6 seeds on Mar-Apr & Jun)
  Step 3: 04_evaluate_heldout_and_stress.py (Frozen 154-day held-out DES + ablations + 30-rep + stress + Device_ID)
  Step 4: 05_statistical_benchmark_report.py (7-day block bootstrap + Wilcoxon + Side-by-Side Comparison Report)
"""

import os
import subprocess
import sys
import time

ALGO_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Step 1/4: Chronological Twin Calibration (Jan-Feb & June)", "02_calibrate_twin_parameters.py"),
    ("Step 2/4: From-Scratch Metaheuristic Policy Optimization (112 evals/seed)", "03_optimize_policies.py"),
    ("Step 3/4: Frozen 154-Day Held-Out DES, 30-Rep Resampling, Stress & Device Audit", "04_evaluate_heldout_and_stress.py"),
    ("Step 4/4: 7-Day Block Bootstrap Inference & Comparison Report", "05_statistical_benchmark_report.py"),
]


def main():
    t_total = time.time()
    print("=" * 88)
    print("JMIR 2026 INDEPENDENT FROM-SCRATCH 10-MONTH RERUN (ZERO INHERITED RESULTS)")
    print("=" * 88)
    for title, script in STEPS:
        print(f"\n>>> Running {title} ({script})...")
        t0 = time.time()
        cmd = [sys.executable, "-u", os.path.join(ALGO_DIR, script)]
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONPATH"] = ALGO_DIR + os.pathsep + env.get("PYTHONPATH", "")
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            print(f"[ERROR] {script} exited with code {ret.returncode}")
            sys.exit(ret.returncode)
        print(f"<<< Finished {script} in {time.time() - t0:.2f}s")
    print("\n" + "=" * 88)
    print(f"COMPLETE PIPELINE FINISHED FROM SCRATCH IN {time.time() - t_total:.2f}s")
    print("Inspect `outputs/COMPARISON_VS_MANUSCRIPT.md` for full side-by-side comparison.")
    print("=" * 88)


if __name__ == "__main__":
    main()
