"""Statistical significance testing (muc 1.5.5): Welch's t-test (unequal
variance) as the default, falling back to Mann-Whitney U when a Shapiro-Wilk
normality check rejects normality for either sample; Bonferroni correction
applied when comparing GNN-MAPPO against multiple baselines simultaneously.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def compare_two_samples(a: list[float], b: list[float], alpha: float = 0.05) -> dict:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)

    # Welch's t-test / Mann-Whitney U and Cohen's d are undefined (or wildly
    # misleading, e.g. a near-zero pooled std blowing up Cohen's d) with fewer
    # than 2 samples per side — this happens for single-seed heuristics
    # (Random Walk / Greedy, muc 1.4.3 uses them only to sanity-check the
    # pipeline in GD1, not for seed-based statistics). Report "insufficient
    # data" instead of a spurious number in that case.
    if len(a) < 2 or len(b) < 2:
        return {
            "test": "insufficient_data", "statistic": None, "p_value": None,
            "significant": False,
            "mean_a": float(a.mean()) if len(a) else None,
            "mean_b": float(b.mean()) if len(b) else None,
            "cohens_d": None,
        }

    normal_a = stats.shapiro(a).pvalue > 0.05 if len(a) >= 3 else False
    normal_b = stats.shapiro(b).pvalue > 0.05 if len(b) >= 3 else False

    if normal_a and normal_b:
        stat, p = stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test
        test_name = "welch_t"
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_name = "mann_whitney_u"

    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    cohens_d = float((a.mean() - b.mean()) / max(pooled_std, 1e-8))

    return {
        "test": test_name, "statistic": float(stat), "p_value": float(p),
        "significant": bool(p < alpha), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "cohens_d": cohens_d,
    }


def compare_against_baselines(target_samples: list[float], baselines: dict[str, list[float]],
                               alpha: float = 0.05) -> dict:
    """Bonferroni-corrected comparison of `target_samples` (e.g. GNN-MAPPO)
    against each baseline in `baselines` (name -> list of per-seed values)."""
    n_comparisons = max(len(baselines), 1)
    corrected_alpha = alpha / n_comparisons
    results = {}
    for name, values in baselines.items():
        r = compare_two_samples(target_samples, values, alpha=corrected_alpha)
        r["alpha_corrected"] = corrected_alpha
        r["significant_bonferroni"] = r["p_value"] is not None and r["p_value"] < corrected_alpha
        results[name] = r
    return results
