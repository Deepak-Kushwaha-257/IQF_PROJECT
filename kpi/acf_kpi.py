"""
kpi/acf_kpi.py
==============
Autocorrelation (ACF) KPI (Section 4.3.4):

Key trick: collect all lag-k pairs across ALL test windows (not just within one window).
This gives enough pairs to compute reliable ACF even with q=10.

ACF score uses Fisher Z-transformation to convert to probability space.
  ACF(ρ1, ρ2) = 1 - Φ( |ρ1 - ρ2| / sqrt(1/(n1-3) + 1/(n2-3)) )

Computed for f(x) = x, f(x) = |x|, f(x) = x²
Paper uses f(x) and f(x²) for the composite score.
"""

import numpy as np
from scipy import stats


# ──────────────────────────────────────────────
# ACF helpers
# ──────────────────────────────────────────────
def collect_lag_pairs(X: np.ndarray, lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect all lag-k pairs from a 3D array of windows.

    Parameters
    ----------
    X   : np.ndarray, shape (N, q, d) — N windows, q timesteps, d series
    lag : int

    Returns
    -------
    pairs_t  : (N*(q-lag), d) — values at time t
    pairs_tl : (N*(q-lag), d) — values at time t+lag
    """
    N, q, d = X.shape
    t_idx  = np.arange(q - lag)        # [0, 1, ..., q-lag-1]
    tl_idx = t_idx + lag               # [lag, lag+1, ..., q-1]

    pairs_t  = X[:, t_idx, :].reshape(-1, d)   # (N*(q-lag), d)
    pairs_tl = X[:, tl_idx, :].reshape(-1, d)

    return pairs_t, pairs_tl


def compute_acf(X: np.ndarray, lag: int = 1, transform=None) -> np.ndarray:
    """
    Compute ACF at a given lag for each of d series.

    Parameters
    ----------
    X         : np.ndarray, shape (N, q, d)
    lag       : int
    transform : callable or None — e.g. np.abs, np.square

    Returns
    -------
    acf : np.ndarray, shape (d,) — correlation at given lag per series
    """
    pairs_t, pairs_tl = collect_lag_pairs(X, lag)
    if transform is not None:
        pairs_t  = transform(pairs_t)
        pairs_tl = transform(pairs_tl)

    d = X.shape[-1]
    acf = np.zeros(d)
    for j in range(d):
        corr = np.corrcoef(pairs_t[:, j], pairs_tl[:, j])[0, 1]
        acf[j] = corr if not np.isnan(corr) else 0.0
    return acf


# ──────────────────────────────────────────────
# Fisher Z-transformation
# ──────────────────────────────────────────────
def fisher_z_pvalue(rho1: float, rho2: float, n1: int, n2: int) -> float:
    """
    Fisher Z-transformation test for equality of two correlation coefficients.
    Eq. 46 in paper.

    Returns p-value: ACF(ρ1, ρ2) = 1 - Φ(|ρ1-ρ2| / sqrt(1/(n1-3)+1/(n2-3)))
    """
    # Clip to avoid arctanh(±1)
    rho1 = np.clip(rho1, -0.9999, 0.9999)
    rho2 = np.clip(rho2, -0.9999, 0.9999)

    z1 = np.arctanh(rho1)
    z2 = np.arctanh(rho2)
    se = np.sqrt(1 / max(n1 - 3, 1) + 1 / max(n2 - 3, 1))
    z_stat = abs(z1 - z2) / se

    # Two-tailed p-value
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    # ACF score = 1 - p_value  (lower = better, consistent with paper)
    return float(1 - p_val)


# ──────────────────────────────────────────────
# Main ACF KPI function
# ──────────────────────────────────────────────
def compute_acf_kpi(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    lags: list = None,
    tenors: list = None,
) -> dict:
    """
    Compute ACF KPI comparing real vs synthetic windows.

    Parameters
    ----------
    X_real  : (N_test, q, d)
    X_synth : (N_test, n_samples, q, d)
    lags    : list of lags to evaluate (default [1, 2])
    tenors  : list of tenor names

    Returns
    -------
    dict with ACF scores per tenor, plus summary ACF score
    """
    if lags is None:
        lags = [1, 2]

    N_test, n_samples, q, d = X_synth.shape
    if tenors is None:
        tenors = [f"t{i}" for i in range(d)]

    # For synthetic: average across the n_samples dimension first
    X_synth_avg = X_synth.mean(axis=1)   # (N_test, q, d)

    n_real  = N_test * (q - max(lags))
    n_synth = N_test * (q - max(lags))

    results = {}
    acf_scores_all = []

    for j, tenor in enumerate(tenors):
        tenor_scores = []
        tenor_results = {}

        for transform, fname in [(None, "x"), (np.abs, "abs_x"), (np.square, "x2")]:
            for lag in lags:
                acf_real  = compute_acf(X_real,      lag, transform)[j]
                acf_synth = compute_acf(X_synth_avg, lag, transform)[j]

                # Fisher Z p-value based ACF score
                score = fisher_z_pvalue(acf_real, acf_synth, n_real, n_synth)
                key = f"{fname}_lag{lag}"
                tenor_results[key] = {
                    "acf_real":  acf_real,
                    "acf_synth": acf_synth,
                    "score":     score,
                }
                if fname in ("x", "x2"):  # paper uses x and x² for composite
                    tenor_scores.append(score)

        # ACF score per tenor = average of x and x² scores
        avg_score = np.mean(tenor_scores) if tenor_scores else 0.5
        tenor_results["acf_score"] = avg_score
        results[tenor] = tenor_results
        acf_scores_all.append(avg_score)

    # Summary ACF score = mean across tenors
    results["__summary__"] = {
        "ACF": np.mean(acf_scores_all)
    }

    return results


def print_acf_kpi(results: dict, model_name: str = "Model"):
    summary = results.get("__summary__", {})
    print(f"\n{'='*50}")
    print(f"ACF KPI — {model_name}")
    print(f"{'='*50}")
    print(f"  ACF score (avg Fisher Z): {summary.get('ACF', 'N/A'):.4f}")
    print()
    for tenor, v in results.items():
        if tenor.startswith("__"):
            continue
        print(f"  {tenor:5s}  score={v.get('acf_score', 'N/A'):.4f}  |  ", end="")
        for lag in [1, 2]:
            if f"x_lag{lag}" in v:
                r = v[f"x_lag{lag}"]["acf_real"]
                s = v[f"x_lag{lag}"]["acf_synth"]
                print(f"ACF_lag{lag}: real={r:.3f} synth={s:.3f}  ", end="")
        print()
