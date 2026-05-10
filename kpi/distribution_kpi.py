"""
kpi/distribution_kpi.py
========================
Distribution distance KPIs (Section 4.3.2):

  1. Earth Mover Distance (EMD / Wasserstein-1)
  2. DY metric (Dragulescu & Yakovenko 2002)
  3. Kolmogorov-Smirnov (KS) test on sample moments
  4. KS test on raw returns (series distance)

All metrics compare synthetic vs real data distributions.
Lower = better (closer to real distribution).
"""

import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance


# ──────────────────────────────────────────────
# Sample Moments
# ──────────────────────────────────────────────
def compute_sample_moments(X: np.ndarray) -> dict:
    """
    Compute per-window sample moments along the time axis.

    Parameters
    ----------
    X : np.ndarray, shape (N, q, d)

    Returns
    -------
    dict with keys 'mean', 'std' — each shape (N, d)
    """
    return {
        "mean": X.mean(axis=1),   # (N, d)
        "std":  X.std(axis=1),    # (N, d)
    }


# ──────────────────────────────────────────────
# EMD
# ──────────────────────────────────────────────
def emd_score(real: np.ndarray, synth: np.ndarray) -> float:
    """
    Earth Mover Distance (Wasserstein-1) between two 1-D arrays.
    Uses scipy.stats.wasserstein_distance (Eq. 44 in paper).
    """
    return float(wasserstein_distance(real, synth))


# ──────────────────────────────────────────────
# DY metric
# ──────────────────────────────────────────────
def dy_metric(real: np.ndarray, synth: np.ndarray, n_bins: int = 50) -> float:
    """
    DY metric (Dragulescu & Yakovenko 2002), Eq. 45:
    DY = Σ_x | log P_r(A_x) - log P_g(A_x) |

    Uses equal-frequency binning on the real distribution.
    """
    # Build bin edges using real data quantiles (equal frequency)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(real, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return np.nan

    counts_r, _ = np.histogram(real,  bins=bin_edges, density=False)
    counts_s, _ = np.histogram(synth, bins=bin_edges, density=False)

    # Convert to densities (avoid log(0))
    p_r = (counts_r + 1e-8) / (counts_r.sum() + 1e-8 * len(counts_r))
    p_s = (counts_s + 1e-8) / (counts_s.sum() + 1e-8 * len(counts_s))

    return float(np.sum(np.abs(np.log(p_r) - np.log(p_s))))


# ──────────────────────────────────────────────
# KS test
# ──────────────────────────────────────────────
def ks_test(real: np.ndarray, synth: np.ndarray) -> tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test.
    Returns (ks_statistic, p_value).
    """
    result = stats.ks_2samp(real, synth)
    return float(result.statistic), float(result.pvalue)


# ──────────────────────────────────────────────
# Main distribution KPI function
# ──────────────────────────────────────────────
def compute_distribution_kpi(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    tenors: list = None,
) -> dict:
    """
    Compute all distribution KPIs comparing real vs synthetic windows.

    Parameters
    ----------
    X_real  : np.ndarray, shape (N_test, q, d)
    X_synth : np.ndarray, shape (N_test, n_samples, q, d)
              For each test window, n_samples synthetic paths
    tenors  : list of tenor names (optional, for labeling)

    Returns
    -------
    dict with keys per tenor, containing EMD, DY, KS, 1-KSpval, KSpval
    for sample moments (mean, std) and raw returns.
    """
    N_test, n_samples, q, d = X_synth.shape
    if tenors is None:
        tenors = [f"t{i}" for i in range(d)]

    results = {}

    for j, tenor in enumerate(tenors):
        real_j  = X_real[:, :, j]    # (N_test, q)
        synth_j = X_synth[:, :, :, j]  # (N_test, n_samples, q)

        # ── Sample moments ──
        real_mean  = real_j.mean(axis=1)        # (N_test,)
        real_std   = real_j.std(axis=1)
        # For synthetic: average over samples per test window
        synth_mean = synth_j.mean(axis=2).mean(axis=1)   # (N_test,)
        synth_std  = synth_j.std(axis=2).mean(axis=1)

        ks_mean_stat, ks_mean_pval = ks_test(real_mean, synth_mean)
        ks_std_stat,  ks_std_pval  = ks_test(real_std, synth_std)
        emd_mean = emd_score(real_mean, synth_mean)
        emd_std  = emd_score(real_std, synth_std)
        dy_mean  = dy_metric(real_mean, synth_mean)

        # ── Raw returns (series distance) ──
        real_flat  = real_j.flatten()
        synth_flat = synth_j.reshape(-1)
        ks_ret_stat, ks_ret_pval = ks_test(real_flat, synth_flat)

        # Distribution score = 1 - KSpval (lower = better, paper convention)
        dist_score = np.mean([1 - ks_mean_pval, 1 - ks_std_pval])

        results[tenor] = {
            "emd_mean":   emd_mean,
            "emd_std":    emd_std,
            "dy_mean":    dy_mean,
            "ks_mean":    ks_mean_stat,
            "ks_mean_pval": ks_mean_pval,
            "ks_std":     ks_std_stat,
            "ks_std_pval":  ks_std_pval,
            "ks_ret":     ks_ret_stat,
            "ks_ret_pval":  ks_ret_pval,
            "dist_score": dist_score,         # 1-KSpval for moments
            "series_dist": 1 - ks_ret_pval,  # 1-KSpval for raw returns
        }

    # Aggregate DIST score across tenors
    dist_scores   = [results[t]["dist_score"]  for t in tenors]
    series_dists  = [results[t]["series_dist"] for t in tenors]
    results["__summary__"] = {
        "DIST":        np.mean(dist_scores),
        "SERIES_DIST": np.mean(series_dists),
    }

    return results


def print_distribution_kpi(results: dict, model_name: str = "Model"):
    summary = results.get("__summary__", {})
    print(f"\n{'='*50}")
    print(f"Distribution KPI — {model_name}")
    print(f"{'='*50}")
    print(f"  DIST score (avg 1-KSpval moments): {summary.get('DIST', 'N/A'):.4f}")
    print(f"  Series DIST (avg 1-KSpval returns): {summary.get('SERIES_DIST', 'N/A'):.4f}")
    print()
    for tenor, v in results.items():
        if tenor.startswith("__"):
            continue
        print(f"  {tenor:5s}  EMD={v['emd_mean']:.5f}  "
              f"KS_mean={v['ks_mean']:.3f}(p={v['ks_mean_pval']:.3f})  "
              f"KS_ret={v['ks_ret']:.3f}(p={v['ks_ret_pval']:.3f})")
