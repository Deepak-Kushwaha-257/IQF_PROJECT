"""
utils/simulate_data.py
=======================
Generate synthetic bivariate time series for model testing.
  - AR(1)+GARCH(1,1) with Normal, t(5), t(3) innovations
  - Cox-Ingersoll-Ross (CIR) model for interest rate levels

All parameters match Table 8 / Section 3.2 of the paper exactly.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal


# ──────────────────────────────────────────────
# GARCH Simulation
# ──────────────────────────────────────────────
@dataclass
class GARCHParams:
    phi1:  float   # AR(1) coefficient
    omega: float   # GARCH omega
    alpha: float   # GARCH alpha
    beta:  float   # GARCH beta
    sigma0: float  # initial vol
    R0:    float   # initial level


# Paper Table 8 parameters
GARCH_PARAMS = {
    "3m": GARCHParams(phi1=0.5,  omega=0.000009, alpha=0.1742, beta=0.8158,
                      sigma0=0.029, R0=0.02),
    "1y": GARCHParams(phi1=-0.5, omega=0.000012, alpha=0.0724, beta=0.9176,
                      sigma0=0.034, R0=0.03),
}
GARCH_CORR = 0.70   # Correlation between innovations


def simulate_garch(
    n_days: int = 7500,
    dist: Literal["normal", "t5", "t3"] = "normal",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate bivariate AR(1)+GARCH(1,1) time series.

    Parameters
    ----------
    n_days : int     — number of daily observations (7500 ≈ 30 years)
    dist   : str     — innovation distribution: 'normal', 't5', 't3'
    seed   : int     — random seed

    Returns
    -------
    DataFrame with columns ['3m', '1y'] containing *level* series,
    and DataFrame with columns ['3m_ret', '1y_ret'] containing returns.
    """
    rng = np.random.default_rng(seed)
    p = GARCH_PARAMS

    # Correlation matrix for bivariate normal base
    corr_matrix = np.array([[1.0, GARCH_CORR], [GARCH_CORR, 1.0]])
    L = np.linalg.cholesky(corr_matrix)  # Cholesky factor

    # Storage
    R   = np.zeros((n_days, 2))     # levels
    x   = np.zeros((n_days, 2))     # returns
    eps = np.zeros((n_days, 2))     # innovations
    sig = np.zeros((n_days, 2))     # conditional volatility

    # Initial conditions
    R[0] = [p["3m"].R0, p["1y"].R0]
    sig[0] = [p["3m"].sigma0, p["1y"].sigma0]

    tenors = ["3m", "1y"]

    for t in range(1, n_days):
        # Draw correlated standard normals
        z_indep = rng.standard_normal(2)
        z = L @ z_indep   # correlated

        # Apply t-distribution scaling if needed
        if dist == "t5":
            chi2 = rng.chisquare(5, size=2)
            z = z * np.sqrt(5 / chi2)
        elif dist == "t3":
            chi2 = rng.chisquare(3, size=2)
            z = z * np.sqrt(3 / chi2)

        for i, tenor in enumerate(tenors):
            pp = p[tenor]
            # GARCH variance update
            var_t = pp.omega + pp.alpha * eps[t-1, i]**2 + pp.beta * sig[t-1, i]**2
            sig[t, i] = np.sqrt(max(var_t, 1e-12))

            # AR(1) return
            eps[t, i] = sig[t, i] * z[i]
            x[t, i]   = pp.phi1 * x[t-1, i] + eps[t, i]

            # Level
            R[t, i] = R[t-1, i] + x[t, i]

    dates = pd.date_range("2000-01-01", periods=n_days, freq="B")
    levels  = pd.DataFrame(R,   columns=["3m", "1y"], index=dates)
    returns = pd.DataFrame(x,   columns=["3m_ret", "1y_ret"], index=dates)
    returns = returns.iloc[1:]   # drop first row (t=0 has no prior)

    return levels, returns


# ──────────────────────────────────────────────
# CIR Simulation
# ──────────────────────────────────────────────
@dataclass
class CIRParams:
    kappa: float   # mean-reversion speed
    theta: float   # long-run mean
    sigma: float   # volatility


# Paper Section 3.2.2 parameters
CIR_PARAMS = {
    "3m": CIRParams(kappa=0.45, theta=0.02, sigma=0.15),
    "1y": CIRParams(kappa=0.20, theta=0.03, sigma=0.10),
}
CIR_CORR = 0.60


def simulate_cir(
    n_days: int = 7500,
    dt: float = 1/252,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate bivariate CIR (Cox-Ingersoll-Ross) interest rate model.
    Uses Euler discretization with correlated Brownian motions.

    dR_t = kappa*(theta - R_t)*dt + sigma*sqrt(R_t)*dW_t

    Returns
    -------
    levels, returns — both as DataFrames with columns ['3m', '1y']
    """
    rng = np.random.default_rng(seed)
    corr_matrix = np.array([[1.0, CIR_CORR], [CIR_CORR, 1.0]])
    L = np.linalg.cholesky(corr_matrix)

    R = np.zeros((n_days, 2))
    p = [CIR_PARAMS["3m"], CIR_PARAMS["1y"]]

    # Initial levels at long-run mean
    R[0] = [p[0].theta, p[1].theta]

    sqrt_dt = np.sqrt(dt)
    for t in range(1, n_days):
        z_indep = rng.standard_normal(2)
        dW = L @ z_indep * sqrt_dt

        for i in range(2):
            drift    = p[i].kappa * (p[i].theta - R[t-1, i]) * dt
            diffusion = p[i].sigma * np.sqrt(max(R[t-1, i], 0)) * dW[i]
            R[t, i] = max(R[t-1, i] + drift + diffusion, 1e-6)  # keep positive

    dates   = pd.date_range("2000-01-01", periods=n_days, freq="B")
    levels  = pd.DataFrame(R, columns=["3m", "1y"], index=dates)
    returns = levels.diff().dropna()
    returns.columns = ["3m_ret", "1y_ret"]

    return levels, returns


# ──────────────────────────────────────────────
# Generate all simulation datasets
# ──────────────────────────────────────────────
def generate_all_simulated(out_dir: str = "data/simulated", n_paths: int = 5):
    """
    Generate 5 paths × 4 DGPs = 20 datasets and save to disk.
    Matches Section 4.4 of the paper.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    configs = [
        ("garch_normal", lambda seed: simulate_garch(dist="normal", seed=seed)),
        ("garch_t5",     lambda seed: simulate_garch(dist="t5",     seed=seed)),
        ("garch_t3",     lambda seed: simulate_garch(dist="t3",     seed=seed)),
        ("cir",          lambda seed: simulate_cir(seed=seed)),
    ]

    for dgp_name, sim_fn in configs:
        for path_idx in range(1, n_paths + 1):
            seed = 1000 * (path_idx) + hash(dgp_name) % 1000
            levels, returns = sim_fn(seed=seed)

            fname_base = f"{dgp_name}_path{path_idx}"
            levels.to_csv(os.path.join(out_dir, f"{fname_base}_levels.csv"))
            returns.to_csv(os.path.join(out_dir, f"{fname_base}_returns.csv"))
            print(f"Generated {fname_base}: {len(levels)} days")

    print(f"\nAll simulated datasets saved to {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/simulated")
    parser.add_argument("--n_paths", type=int, default=5)
    args = parser.parse_args()
    generate_all_simulated(args.out_dir, args.n_paths)
