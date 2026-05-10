"""
kpi/backtesting_kpi.py
======================
VaR Backtesting KPIs (Section 4.3.6):

Uses Probability Integral Transform (PIT):
  u_t = F̂_t(x_t)  where F̂_t is the synthetic forecast distribution

u-values should follow Uniform[0,1] if model is correctly specified.

KPIs:
  - KS test of u-values vs Uniform
  - u-value histogram (DIFF = avg deviation from 1.0)
  - Breach rates at 2.5%, 5%, 10%, 90%, 95%, 97.5%
  - BT score = [BR + (1 - KSpval)] / 2

Paper reference: Section 4.3.6, Crnkovic & Drachman (1996), Berkowitz (2001)
"""

import numpy as np
from scipy import stats


BREACH_LEVELS = [0.025, 0.05, 0.10, 0.90, 0.95, 0.975]
N_HIST_BINS   = 10


# ──────────────────────────────────────────────
# u-value computation
# ──────────────────────────────────────────────
def compute_u_values(
    real_returns: np.ndarray,
    synth_paths: np.ndarray,
    step: int = 0,
) -> np.ndarray:
    """
    Compute u-values for a single tenor over multiple test dates.
    u_t = F̂_t(x_t) = (fraction of synthetic paths < x_t)

    Parameters
    ----------
    real_returns : np.ndarray, shape (N_test,) — realized 1-day returns per test date
    synth_paths  : np.ndarray, shape (N_test, n_samples) — synthetic forecast distribution
    step         : int — which forecast step to use (0 = 1-day ahead)

    Returns
    -------
    u_values : np.ndarray, shape (N_test,)
    """
    N = len(real_returns)
    u_values = np.zeros(N)

    for t in range(N):
        x_t = real_returns[t]
        synth_t = synth_paths[t]   # (n_samples,)
        u_values[t] = np.mean(synth_t < x_t)  # empirical CDF

    return u_values


# ──────────────────────────────────────────────
# Breach rates
# ──────────────────────────────────────────────
def compute_breach_rates(u_values: np.ndarray) -> dict:
    """
    Compute absolute deviation between actual and expected breach rates.
    Lower = better (model is correctly calibrated).
    """
    br = {}
    for p in BREACH_LEVELS:
        if p < 0.5:
            # Left tail: fraction below p-quantile (should be p)
            actual = np.mean(u_values < p)
        else:
            # Right tail: fraction above p-quantile (should be 1-p)
            actual = np.mean(u_values > p)
        br[f"BR{int(p*1000):04d}"] = abs(actual - (p if p < 0.5 else 1-p))

    # Summed breach rate (paper formula)
    br["BR_sum"] = sum(br.values())
    return br


# ──────────────────────────────────────────────
# u-value histogram DIFF
# ──────────────────────────────────────────────
def histogram_diff(u_values: np.ndarray) -> float:
    """
    Average absolute deviation of u-value histogram bar heights from 1.0.
    (Under uniformity all bars = 1.0)
    """
    counts, _ = np.histogram(u_values, bins=N_HIST_BINS, range=(0, 1))
    n = len(u_values)
    heights = (counts / n) * N_HIST_BINS   # normalize so uniform = 1.0
    return float(np.mean(np.abs(heights - 1.0)))


# ──────────────────────────────────────────────
# Main backtest KPI function
# ──────────────────────────────────────────────
def compute_backtesting_kpi(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    tenors: list = None,
    n_subperiods: int = 3,
    forecast_step: int = 0,   # 0 = 1-day ahead
) -> dict:
    """
    Full backtesting KPI computation.

    Parameters
    ----------
    X_real   : (N_test, q, d)
    X_synth  : (N_test, n_samples, q, d)
    tenors   : list of tenor names
    n_subperiods : number of sub-periods for sub-period analysis
    forecast_step: which timestep in q to evaluate (0 = 1-day ahead)

    Returns
    -------
    dict with BT scores per tenor, sub-period analysis, and summary BT score
    """
    N_test, n_samples, q, d = X_synth.shape
    if tenors is None:
        tenors = [f"t{i}" for i in range(d)]

    results = {}
    bt_scores_all = []

    for j, tenor in enumerate(tenors):
        # Realized 1-day returns
        real_j  = X_real[:, forecast_step, j]         # (N_test,)
        synth_j = X_synth[:, :, forecast_step, j]     # (N_test, n_samples)

        # u-values
        u_vals = compute_u_values(real_j, synth_j)

        # KS test vs Uniform
        ks_stat, ks_pval = stats.kstest(u_vals, "uniform")

        # Histogram DIFF
        diff = histogram_diff(u_vals)

        # Breach rates
        br = compute_breach_rates(u_vals)

        # BT score = [BR_sum + (1 - KSpval)] / 2  (Eq. in paper Section 4.3.7)
        bt_score = (br["BR_sum"] + (1 - ks_pval)) / 2

        # Sub-period analysis (split into n_subperiods)
        sub_bt_scores = []
        if N_test >= n_subperiods * 10:
            sub_size = N_test // n_subperiods
            for s in range(n_subperiods):
                start = s * sub_size
                end   = (s + 1) * sub_size if s < n_subperiods - 1 else N_test
                u_sub = u_vals[start:end]
                if len(u_sub) < 10:
                    continue
                _, ksp_sub = stats.kstest(u_sub, "uniform")
                br_sub = compute_breach_rates(u_sub)
                sub_bt = (br_sub["BR_sum"] + (1 - ksp_sub)) / 2
                sub_bt_scores.append(sub_bt)

        # Median across sub-periods (or full period if sub-period failed)
        bt_score_final = np.median(sub_bt_scores) if sub_bt_scores else bt_score

        tenor_result = {
            "u_values": u_vals,
            "ks_stat":  ks_stat,
            "ks_pval":  ks_pval,
            "diff":     diff,
            "bt_score": bt_score_final,
            **br,
        }
        results[tenor] = tenor_result
        bt_scores_all.append(bt_score_final)

    # Summary BT score = median across tenors, averaged for 1-day and 10-day
    results["__summary__"] = {
        "BT": np.median(bt_scores_all)
    }

    return results


def print_backtesting_kpi(results: dict, model_name: str = "Model"):
    summary = results.get("__summary__", {})
    print(f"\n{'='*50}")
    print(f"Backtesting KPI — {model_name}")
    print(f"{'='*50}")
    print(f"  BT score: {summary.get('BT', 'N/A'):.4f}")
    print()
    for tenor, v in results.items():
        if tenor.startswith("__"):
            continue
        print(f"  {tenor:5s}  BT={v['bt_score']:.4f}  "
              f"KSpval={v['ks_pval']:.4f}  DIFF={v['diff']:.4f}  "
              f"BR05={v.get('BR0050', 0):.4f}  BR95={v.get('BR0950', 0):.4f}")
