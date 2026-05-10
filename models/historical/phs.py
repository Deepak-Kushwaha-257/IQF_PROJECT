"""
models/historical/phs.py
========================
Plain Historical Simulation (PHS).

For each forecast date t0:
  - Look back 251 business days of returns
  - Sample those returns randomly (with replacement) to form forecast paths
  - For multi-step, use consecutive historical blocks (Table 4 in paper)

Paper reference: Section 2.1.1
"""

import numpy as np
import joblib
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from models.base_model import BaseModel


class PHS(BaseModel):
    """
    Plain Historical Simulation.

    Stores all training return windows. At generation time, draws
    random windows (with replacement) from the historical pool.
    This replicates the empirical distribution approach of the paper.
    """

    def __init__(self, p: int = 10, q: int = 10, d: int = 9, window: int = 251):
        super().__init__("PHS", p, q, d)
        self.window = window          # historical lookback (251 = 1 trading year)
        self.historical_targets = None  # (N_train, q, d) — unscaled train targets

    def fit(self, X_cond_train: np.ndarray, X_tgt_train: np.ndarray) -> "PHS":
        """
        Store historical return windows.
        PHS requires no actual model fitting — just stores the training returns.
        """
        self.historical_targets = X_tgt_train.copy()   # (N_train, q, d)
        self.is_fitted = True
        print(f"PHS fitted: {len(self.historical_targets)} historical windows stored.")
        return self

    def generate(self, condition: np.ndarray, n_samples: int = 251) -> np.ndarray:
        """
        Draw n_samples random historical windows as forecast paths.

        Parameters
        ----------
        condition : np.ndarray, shape (p, d) — not used by PHS (non-conditional)
        n_samples : int

        Returns
        -------
        np.ndarray, shape (n_samples, q, d)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before generate()")

        N = len(self.historical_targets)
        idx = np.random.choice(N, size=n_samples, replace=True)
        return self.historical_targets[idx]    # (n_samples, q, d)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({"historical_targets": self.historical_targets,
                     "p": self.p, "q": self.q, "d": self.d}, path)

    def load(self, path: str) -> "PHS":
        data = joblib.load(path)
        self.historical_targets = data["historical_targets"]
        self.p = data["p"]
        self.q = data["q"]
        self.d = data["d"]
        self.is_fitted = True
        return self
