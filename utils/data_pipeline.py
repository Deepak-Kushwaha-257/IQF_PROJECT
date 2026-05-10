"""
utils/data_pipeline.py
======================
Complete data pipeline:
  1. Load raw FRED CSV or DataFrame
  2. Compute absolute returns (rate changes)
  3. Handle missing values
  4. Standardize (fit on train, transform train+test)
  5. Create rolling windows of shape (n_windows, p+q, d)
  6. Train/test split
  7. Save / load processed arrays
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import yaml


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TENORS = ["3m", "6m", "1y", "2y", "3y", "5y", "10y", "20y", "30y"]
FRED_TICKERS = {
    "DGS3MO": "3m",
    "DGS6MO": "6m",
    "DGS1":   "1y",
    "DGS2":   "2y",
    "DGS3":   "3y",
    "DGS5":   "5y",
    "DGS10":  "10y",
    "DGS20":  "20y",
    "DGS30":  "30y",
}


# ──────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────
def load_fred_csv(data_dir: str) -> pd.DataFrame:
    """
    Load pre-downloaded FRED CSVs from data/raw/.
    Each file is named by FRED ticker (e.g. DGS3MO.csv).
    Returns a DataFrame with columns = tenor names, index = date.
    """
    dfs = []
    for ticker, name in FRED_TICKERS.items():
        path = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"Run:  python scripts/download_fred_data.py"
            )
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.rename(columns={df.columns[0]: name})
        dfs.append(df)
    combined = pd.concat(dfs, axis=1)
    combined.index.name = "date"
    combined = combined.sort_index()
    return combined


def load_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Accept a pre-built DataFrame with 9 tenor columns."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
def compute_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """
    Compute absolute returns: x_t = R_t - R_{t-1}
    For interest rates we use absolute (not log) returns per the paper.
    """
    returns = levels.diff().dropna()
    return returns


def handle_missing(df: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """
    Forward-fill gaps up to max_gap days (holidays), then drop remaining NaNs.
    """
    # Replace '.' (FRED missing marker) with NaN
    df = df.replace(".", np.nan).astype(float)
    # Forward fill short gaps (weekends/holidays)
    df = df.fillna(method="ffill", limit=max_gap)
    # Drop rows with any remaining NaN
    df = df.dropna()
    return df


def filter_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[start:end]


# ──────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────
def create_windows(
    returns: np.ndarray,
    p: int = 10,
    q: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of size (p+q) over the return series.

    Parameters
    ----------
    returns : np.ndarray, shape (T, d)
    p       : condition length (past days)
    q       : target length (future days)

    Returns
    -------
    conditions : np.ndarray, shape (N, p, d)
    targets    : np.ndarray, shape (N, q, d)
    """
    T, d = returns.shape
    window = p + q
    n_windows = T - window + 1

    conditions = np.zeros((n_windows, p, d), dtype=np.float32)
    targets    = np.zeros((n_windows, q, d), dtype=np.float32)

    for i in range(n_windows):
        conditions[i] = returns[i : i + p]
        targets[i]    = returns[i + p : i + p + q]

    return conditions, targets


# ──────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────
class DataPipeline:
    """
    End-to-end pipeline: raw levels → windowed arrays ready for models.

    Usage
    -----
    >>> pipeline = DataPipeline(p=10, q=10, train_split=0.8, scaler="standard")
    >>> pipeline.fit_transform(levels_df)
    >>> # Access:
    >>> pipeline.X_train_cond  # (N_train, p, d)
    >>> pipeline.X_train_tgt   # (N_train, q, d)
    >>> pipeline.X_test_cond   # (N_test, p, d)
    >>> pipeline.X_test_tgt    # (N_test, q, d)
    >>> pipeline.returns_raw   # unscaled returns DataFrame
    >>> pipeline.scaler        # fitted scaler for inverse_transform
    """

    def __init__(
        self,
        p: int = 10,
        q: int = 10,
        train_split: float = 0.80,
        scaler: str = "standard",
        random_seed: int = 42,
    ):
        self.p = p
        self.q = q
        self.train_split = train_split
        self.random_seed = random_seed
        self.scaler_type = scaler
        self.scaler = None

        # Outputs
        self.returns_raw = None
        self.returns_scaled = None
        self.X_train_cond = None
        self.X_train_tgt = None
        self.X_test_cond = None
        self.X_test_tgt = None
        self.train_idx = None
        self.test_idx = None

    def fit_transform(self, levels: pd.DataFrame) -> "DataPipeline":
        """Run full pipeline on a levels DataFrame."""
        # Step 1 — Returns
        levels = handle_missing(levels)
        self.returns_raw = compute_returns(levels)

        # Step 2 — Create windows from raw returns (before scaling, for alignment)
        R = self.returns_raw.values.astype(np.float32)
        conditions, targets = create_windows(R, self.p, self.q)

        # Step 3 — Train/test split on windows (random, per paper)
        N = conditions.shape[0]
        np.random.seed(self.random_seed)
        perm = np.random.permutation(N)
        n_train = int(N * self.train_split)
        self.train_idx = np.sort(perm[:n_train])
        self.test_idx  = np.sort(perm[n_train:])

        cond_train = conditions[self.train_idx]
        tgt_train  = targets[self.train_idx]
        cond_test  = conditions[self.test_idx]
        tgt_test   = targets[self.test_idx]

        # Step 4 — Fit scaler on training data only
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))

        # Flatten (N, seq, d) → (N*seq, d) to fit scaler across all timesteps
        d = cond_train.shape[-1]
        flat_train = np.concatenate(
            [cond_train.reshape(-1, d), tgt_train.reshape(-1, d)], axis=0
        )
        self.scaler.fit(flat_train)

        # Scale
        def _scale(arr):
            n, s, dd = arr.shape
            return self.scaler.transform(arr.reshape(-1, dd)).reshape(n, s, dd)

        self.X_train_cond = _scale(cond_train)
        self.X_train_tgt  = _scale(tgt_train)
        self.X_test_cond  = _scale(cond_test)
        self.X_test_tgt   = _scale(tgt_test)

        print(
            f"Pipeline complete:\n"
            f"  Train: {self.X_train_cond.shape[0]} windows  "
            f"| Test: {self.X_test_cond.shape[0]} windows\n"
            f"  Condition shape: {self.X_train_cond.shape[1:]}  "
            f"| Target shape: {self.X_train_tgt.shape[1:]}"
        )
        return self

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Unscale a (N, seq, d) array back to return space."""
        n, s, d = arr.shape
        return self.scaler.inverse_transform(arr.reshape(-1, d)).reshape(n, s, d)

    def save(self, out_dir: str, prefix: str = "usdyc"):
        """Save processed arrays and scaler."""
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{prefix}_train_cond.npy"), self.X_train_cond)
        np.save(os.path.join(out_dir, f"{prefix}_train_tgt.npy"),  self.X_train_tgt)
        np.save(os.path.join(out_dir, f"{prefix}_test_cond.npy"),  self.X_test_cond)
        np.save(os.path.join(out_dir, f"{prefix}_test_tgt.npy"),   self.X_test_tgt)
        joblib.dump(self.scaler, os.path.join(out_dir, f"{prefix}_scaler.pkl"))
        print(f"Saved processed data to {out_dir}/")

    @classmethod
    def load(cls, out_dir: str, prefix: str = "usdyc") -> "DataPipeline":
        """Load previously saved pipeline outputs."""
        obj = cls.__new__(cls)
        obj.X_train_cond = np.load(os.path.join(out_dir, f"{prefix}_train_cond.npy"))
        obj.X_train_tgt  = np.load(os.path.join(out_dir, f"{prefix}_train_tgt.npy"))
        obj.X_test_cond  = np.load(os.path.join(out_dir, f"{prefix}_test_cond.npy"))
        obj.X_test_tgt   = np.load(os.path.join(out_dir, f"{prefix}_test_tgt.npy"))
        obj.scaler       = joblib.load(os.path.join(out_dir, f"{prefix}_scaler.pkl"))
        return obj


# ──────────────────────────────────────────────
# CLI helper
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",  default="data/raw",       help="Raw FRED CSVs dir")
    parser.add_argument("--out_dir",  default="data/processed",  help="Output dir")
    parser.add_argument("--start",    default="2008-01-01")
    parser.add_argument("--end",      default="2023-02-16")
    parser.add_argument("--prefix",   default="usdyc2")
    args = parser.parse_args()

    levels = load_fred_csv(args.raw_dir)
    levels = filter_date_range(levels, args.start, args.end)
    print(f"Loaded {len(levels)} business days, {levels.shape[1]} tenors")
    print(f"Date range: {levels.index[0].date()} → {levels.index[-1].date()}")

    pipeline = DataPipeline(p=10, q=10, train_split=0.8, scaler="standard")
    pipeline.fit_transform(levels)
    pipeline.save(args.out_dir, prefix=args.prefix)
