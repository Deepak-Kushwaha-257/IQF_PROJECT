"""
models/parametric/garch_model.py
=================================
AR(1)+GARCH(1,1) parametric model for conditional distribution forecasting.

Three variants (Table 1 in paper):
  - AR       : AR(1) on yield levels, constant variance
  - AR-RET   : AR(1) on yield returns, constant variance
  - GARCH-RET: AR(1)+GARCH(1,1) on returns, normal errors
  - GARCHt-RET: AR(1)+GARCH(1,1) on returns, t-distributed errors  ← best

Paper reference: Section 2.2.2 and 2.2.3
Uses: arch library (pip install arch)
"""

import numpy as np
import joblib
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from models.base_model import BaseModel

try:
    from arch import arch_model
    from arch.univariate import ARX, GARCH, StudentsT, Normal
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("WARNING: arch library not installed. Run: pip install arch")


class GARCHModel(BaseModel):
    """
    AR(1) + GARCH(1,1) model for each tenor independently.
    Uses Gaussian copula to couple tenors (via Cholesky decomposition of
    historical residual correlation).

    Variants
    --------
    variant = 'ar'       : AR(1) on levels
    variant = 'ar_ret'   : AR(1) on returns
    variant = 'garch'    : AR(1)+GARCH(1,1), normal errors
    variant = 'garch_t'  : AR(1)+GARCH(1,1), t-distribution errors  ← GARCHt-RET
    """

    def __init__(
        self,
        p: int = 10,
        q: int = 10,
        d: int = 9,
        variant: str = "garch_t",
        window_years: int = 3,
        dist: str = "t",
    ):
        name_map = {
            "ar": "AR", "ar_ret": "AR-RET",
            "garch": "GARCH-RET", "garch_t": "GARCHt-RET"
        }
        super().__init__(name_map.get(variant, variant), p, q, d)
        self.variant = variant
        self.window_years = window_years
        self.dist = dist    # 'normal' or 't'

        self.models_   = []    # fitted arch models per tenor
        self.chol_corr = None  # Cholesky of residual correlation matrix
        self.resid_std = None  # standardized residuals from training

    def fit(self, X_cond_train: np.ndarray, X_tgt_train: np.ndarray) -> "GARCHModel":
        if not ARCH_AVAILABLE:
            raise ImportError("arch library required: pip install arch")

        N, q, d = X_tgt_train.shape

        # Flatten all windows into a single return series per tenor
        # Shape: (N*q, d)
        flat_returns = X_tgt_train.reshape(-1, d)
        T = flat_returns.shape[0]

        self.models_   = []
        std_residuals  = np.zeros_like(flat_returns)

        for j in range(d):
            series = flat_returns[:, j]

            if self.variant in ("ar", "ar_ret"):
                # Simple AR(1) — constant variance
                from statsmodels.tsa.ar_model import AutoReg
                ar = AutoReg(series, lags=1, old_names=False).fit()
                resid = ar.resid
                sigma = np.std(resid)
                std_residuals[1:, j] = resid / sigma
                self.models_.append(("ar", ar, sigma))
            else:
                # GARCH(1,1)
                dist_obj = "t" if self.dist == "t" else "normal"
                am = arch_model(series, mean="AR", lags=1,
                                vol="GARCH", p=1, q=1, dist=dist_obj)
                res = am.fit(disp="off", show_warning=False)
                std_residuals[:, j] = res.std_resid
                self.models_.append(("garch", res))

        # Estimate correlation of standardized residuals (Gaussian copula)
        valid = std_residuals[~np.any(np.isnan(std_residuals), axis=1)]
        self.chol_corr = np.linalg.cholesky(
            np.corrcoef(valid.T) + 1e-6 * np.eye(d)
        )

        self.is_fitted = True
        print(f"{self.name} fitted on {T} observations, {d} tenors")
        return self

    def generate(self, condition: np.ndarray, n_samples: int = 251) -> np.ndarray:
        """
        Simulate n_samples paths using fitted GARCH models and Cholesky copula.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before generate()")

        d = self.d
        q = self.q
        paths = np.zeros((n_samples, q, d))

        for s in range(n_samples):
            for step in range(q):
                # Draw correlated standard normals
                z_indep = np.random.standard_normal(d)
                z = self.chol_corr @ z_indep   # correlated

                for j in range(d):
                    model_info = self.models_[j]

                    if model_info[0] == "ar":
                        _, ar_model, sigma = model_info
                        paths[s, step, j] = sigma * z[j]
                    else:
                        _, garch_res = model_info
                        params = garch_res.params

                        # GARCH(1,1) variance forecast
                        if step == 0:
                            # Use last fitted variance as initial
                            last_var = garch_res.conditional_volatility[-1] ** 2
                        else:
                            last_eps = paths[s, step-1, j]
                            omega = params.get("omega", 1e-6)
                            alpha = params.get("alpha[1]", 0.1)
                            beta  = params.get("beta[1]", 0.85)
                            last_var = omega + alpha * last_eps**2 + beta * last_var

                        sigma_t = np.sqrt(max(last_var, 1e-12))

                        if self.dist == "t" and hasattr(garch_res, "params"):
                            nu = garch_res.params.get("nu", 8.0)
                            eps = np.random.standard_t(nu) * sigma_t
                        else:
                            eps = z[j] * sigma_t

                        paths[s, step, j] = eps

        return paths

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "models_": self.models_,
            "chol_corr": self.chol_corr,
            "variant": self.variant,
            "dist": self.dist,
            "p": self.p, "q": self.q, "d": self.d
        }, path)

    def load(self, path: str) -> "GARCHModel":
        data = joblib.load(path)
        self.models_   = data["models_"]
        self.chol_corr = data["chol_corr"]
        self.variant   = data["variant"]
        self.dist      = data["dist"]
        self.p = data["p"]
        self.q = data["q"]
        self.d = data["d"]
        self.is_fitted = True
        return self
