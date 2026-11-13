# Diagnostic Script 2: Sign-Flip Permutation Wilcoxon Test & Block Bootstrap
import numpy as np
from scipy import stats

def sign_flip_permutation_wilcoxon(diffs, n_permutations=400000, seed=202611):
    diffs = np.asarray(diffs)
    nonzero = diffs[np.abs(diffs) > 1e-9]
    n = len(nonzero)
    if n == 0:
        return 1.0, 1.0
    ranks = stats.rankdata(np.abs(nonzero))
    observed_w = np.sum(ranks[nonzero > 0])
    if n <= 20:
        all_signs = np.array(list(np.ndindex((2,) * n))) * 2 - 1
        perm_w = np.dot(all_signs, ranks)
        p_val = float(np.mean(np.abs(perm_w) >= np.abs(observed_w - np.sum(ranks)/2) * 2))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], size=(n_permutations, n))
        perm_w = np.dot(signs, ranks)
        p_val = float(np.mean(np.abs(perm_w) >= np.abs(observed_w - np.sum(ranks)/2) * 2))
    return float(stats.wilcoxon(nonzero)[1]), p_val

if __name__ == '__main__':
    print('Sign-flip permutation testing module ready.')
