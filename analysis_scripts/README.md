# Diagnostic Analysis Scripts

This directory contains standalone audit scripts supporting the diagnostic claims in the manuscript:

1. `01_audit_start_time_invariance.py`: Audits the accounting identity sum(C_j - a_j) = sum(s_j) + sum(p_j) - sum(a_j) and verifies identical start times across all 11 dispatch policies on each of the 154 test days.
2. `02_audit_wilcoxon_permutation.py`: Performs exact and Monte Carlo sign-flip permutation tests for the Wilcoxon signed-rank statistic with zero-difference exclusion and mid-rank tie breaking.
3. `03_audit_calibration_lag_sensitivity.py`: Evaluates the 11 dispatch policies under post-analytic reporting delays drawn from the calibration periods (January–February for Wave 1, June for Wave 2).
