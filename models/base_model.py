"""
models/base_model.py
====================
Abstract base class for all 14 models.
Every model must implement: fit(), generate(), save(), load().
"""

from abc import ABC, abstractmethod
import numpy as np
import os


class BaseModel(ABC):
    """
    Abstract interface for all VaR distribution forecast models.

    All models share the same interface:
        model.fit(X_cond_train, X_tgt_train)
        model.generate(condition, n_samples) -> np.ndarray (n_samples, q, d)
        model.save(path)
        model.load(path)
    """

    def __init__(self, name: str, p: int = 10, q: int = 10, d: int = 9):
        self.name = name
        self.p = p    # condition length
        self.q = q    # forecast length
        self.d = d    # number of risk factors / tenors
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_cond_train: np.ndarray, X_tgt_train: np.ndarray) -> "BaseModel":
        """
        Train model on rolling windows.

        Parameters
        ----------
        X_cond_train : np.ndarray, shape (N_train, p, d)
        X_tgt_train  : np.ndarray, shape (N_train, q, d)
        """
        ...

    @abstractmethod
    def generate(self, condition: np.ndarray, n_samples: int = 251) -> np.ndarray:
        """
        Generate n_samples synthetic forecast paths.

        Parameters
        ----------
        condition : np.ndarray, shape (p, d) — one condition window

        Returns
        -------
        np.ndarray, shape (n_samples, q, d) — synthetic paths
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> "BaseModel":
        ...

    def generate_batch(
        self, X_cond_test: np.ndarray, n_samples: int = 251
    ) -> np.ndarray:
        """
        Convenience: generate for all test conditions.

        Returns
        -------
        np.ndarray, shape (N_test, n_samples, q, d)
        """
        N_test = X_cond_test.shape[0]
        results = []
        for i in range(N_test):
            paths = self.generate(X_cond_test[i], n_samples=n_samples)
            results.append(paths)
        return np.stack(results, axis=0)   # (N_test, n_samples, q, d)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, p={self.p}, q={self.q}, d={self.d})"
