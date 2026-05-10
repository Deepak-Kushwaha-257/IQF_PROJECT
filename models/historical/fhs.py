"""
models/historical/fhs.py
========================
Filtered Historical Simulation (FHS) with EWMA volatility scaling.

Steps (Section 2.1.2):
  1. Estimate EWMA volatility σ_t for each return series
  2. Compute de-volatized returns: x̂_t = x_t / σ_t
  3. Forecast next-day volatility: σ̃_{t+1}
  4. Select 251 historical devol scenarios
  5. Re-volatize: x̃ = x̂_t * σ̃_{t+1}

Paper reference: Section 2.1.2, BARONE-ADESI & GIANNOPOULOS (2001)
"""

import numpy as np
import joblib
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from models.base_model import BaseModel


def ewma_volatility(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """
    Exponentially Weighted Moving Average (EWMA) volatility.
    λ = 0.94 is the RiskMetrics daily decay factor.

    Parameters
    ----------
    returns : np.ndarray, shape (T, d)
    lam     : float — decay factor (0.94 for daily data)

    Returns
    -------
    vol : np.ndarray, shape (T, d)
    """
    T, d = returns.shape
    var  = np.zeros((T, d))
    var[0] = returns[0] ** 2

    for t in range(1, T):
        var[t] = lam * var[t-1] + (1 - lam) * returns[t-1] ** 2

    return np.sqrt(np.maximum(var, 1e-12))


class FHS(BaseModel):
    """
    Filtered Historical Simulation using EWMA volatility scaling.
    """

    def __init__(
        self,
        p: int = 10,
        q: int = 10,
        d: int = 9,
        lam: float = 0.94,
        window: int = 251,
    ):
        super().__init__("FHS", p, q, d)
        self.lam = lam
        self.window = window

        # Stored at fit time
        self.devol_targets = None   # (N_train, q, d) de-volatized returns
        self.vol_targets   = None   # (N_train, q, d) volatilities for revol

    def fit(self, X_cond_train: np.ndarray, X_tgt_train: np.ndarray) -> "FHS":
        """
        Compute EWMA vols and de-volatize training returns.
        """
        N, q, d = X_tgt_train.shape

        # Flatten all windows into a single long series for vol estimation
        # Use the target portion only for vol storage
        vol_tgt  = np.zeros_like(X_tgt_train)

        for i in range(N):
            # Estimate vol over the target window
            window_returns = X_tgt_train[i]   # (q, d)
            vol = ewma_volatility(window_returns, self.lam)  #computes risk (sigma_t)
            vol_tgt[i]  = vol

        # De-volatize: x̂ = x / σ
        devol = np.where(vol_tgt > 1e-10, X_tgt_train / vol_tgt, 0.0)

        self.devol_targets = devol
        self.vol_targets   = vol_tgt
        self.is_fitted = True
        print(f"FHS fitted: {N} windows, λ={self.lam}")
        return self

    def generate(self, condition: np.ndarray, n_samples: int = 251) -> np.ndarray:
        """
        Generate n_samples FHS forecast paths.

        1. Estimate forecast vol from condition
        2. Sample devol windows
        3. Re-volatize with forecast vol
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before generate()")

        # Estimate forecast volatility from condition (last step of EWMA)
        cond_vol = ewma_volatility(condition, self.lam)   # (p, d)
        forecast_vol = cond_vol[-1]  # use last day's vol as forecast

        # Sample devol windows
        N = len(self.devol_targets)
        idx = np.random.choice(N, size=n_samples, replace=True)
        sampled_devol = self.devol_targets[idx]   # (n_samples, q, d)

        # Re-volatize: x̃ = x̂ * σ̃_{t+1}
        # Broadcast forecast_vol (d,) → (n_samples, 1, d)
        revol = sampled_devol * forecast_vol[np.newaxis, np.newaxis, :]

        return revol   # (n_samples, q, d)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "devol_targets": self.devol_targets,
            "vol_targets": self.vol_targets,
            "lam": self.lam, "p": self.p, "q": self.q, "d": self.d
        }, path)

    def load(self, path: str) -> "FHS":
        data = joblib.load(path)
        self.devol_targets = data["devol_targets"]
        self.vol_targets   = data["vol_targets"]
        self.lam = data["lam"]
        self.p   = data["p"]
        self.q   = data["q"]
        self.d   = data["d"]
        self.is_fitted = True
        return self
