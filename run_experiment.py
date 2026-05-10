"""
run_experiment.py
=================
Master script to run any subset of models and datasets.

Usage examples:
  # Minimum viable experiment (recommended first run):
  python run_experiment.py --models phs garch_t cwgan cgan_fc --dataset usdyc2

  # All models on one dataset:
  python run_experiment.py --models all --dataset usdyc2

  # Simulated data only:
  python run_experiment.py --models phs garch_t cwgan --dataset garch_normal

  # All datasets + models:
  python run_experiment.py --models all --dataset all
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import yaml
import time
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_pipeline    import DataPipeline, load_fred_csv, filter_date_range
from utils.simulate_data    import simulate_garch, simulate_cir
from kpi.distribution_kpi  import compute_distribution_kpi
from kpi.acf_kpi            import compute_acf_kpi
from kpi.backtesting_kpi    import compute_backtesting_kpi
from kpi.composite_score    import (
    compute_composite_score, build_ranking_table, print_ranking_table
)
from utils.visualization    import plot_distribution_acf, plot_u_value_histograms


# ──────────────────────────────────────────────
# Dataset loader
# ──────────────────────────────────────────────
def load_dataset(dataset_name: str, cfg: dict) -> DataPipeline:
    """Load and preprocess a dataset."""
    print(f"\n[Dataset] Loading {dataset_name}...")

    sim_dists = {"garch_normal": "normal", "garch_t5": "t5", "garch_t3": "t3"}
    p, q = cfg.get("condition_length", 10), cfg.get("target_length", 10)

    if dataset_name in sim_dists:
        # Simulated GARCH data (2 tenors)
        dist = sim_dists[dataset_name]
        levels, returns = simulate_garch(dist=dist, seed=42)
        # Use only returns (2 tenors)
        return_vals = returns[["3m_ret", "1y_ret"]].values.astype(np.float32)
        d = 2
        pipeline = DataPipeline(p=p, q=q, train_split=0.8)
        # Manual fit since we have raw returns
        from sklearn.preprocessing import StandardScaler
        n_windows = len(return_vals) - (p + q) + 1
        conds, tgts = [], []
        for i in range(n_windows):
            conds.append(return_vals[i:i+p])
            tgts.append(return_vals[i+p:i+p+q])
        conds = np.array(conds, dtype=np.float32)
        tgts  = np.array(tgts,  dtype=np.float32)
        N = len(conds)
        np.random.seed(42)
        idx = np.random.permutation(N)
        n_tr = int(N * 0.8)
        tr, te = np.sort(idx[:n_tr]), np.sort(idx[n_tr:])
        scaler = StandardScaler()
        flat = np.concatenate([conds[tr].reshape(-1, d), tgts[tr].reshape(-1, d)])
        scaler.fit(flat)
        def _sc(arr):
            n, s, dd = arr.shape
            return scaler.transform(arr.reshape(-1, dd)).reshape(n, s, dd)
        pipeline.X_train_cond = _sc(conds[tr])
        pipeline.X_train_tgt  = _sc(tgts[tr])
        pipeline.X_test_cond  = _sc(conds[te])
        pipeline.X_test_tgt   = _sc(tgts[te])
        pipeline.scaler       = scaler
        pipeline.d = d
        print(f"  Simulated {dataset_name}: train={len(tr)}, test={len(te)}, tenors={d}")
        return pipeline

    elif dataset_name == "cir":
        levels, returns = simulate_cir(seed=42)
        return_vals = returns[["3m_ret", "1y_ret"]].values.astype(np.float32)
        d = 2
        pipeline = DataPipeline(p=p, q=q)
        # Same manual pipeline as above
        n_windows = len(return_vals) - (p + q) + 1
        conds = np.array([return_vals[i:i+p] for i in range(n_windows)], dtype=np.float32)
        tgts  = np.array([return_vals[i+p:i+p+q] for i in range(n_windows)], dtype=np.float32)
        N = len(conds)
        np.random.seed(42)
        idx = np.random.permutation(N)
        n_tr = int(N * 0.8)
        tr, te = np.sort(idx[:n_tr]), np.sort(idx[n_tr:])
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        flat = np.concatenate([conds[tr].reshape(-1, d), tgts[tr].reshape(-1, d)])
        scaler.fit(flat)
        def _sc(arr):
            n, s, dd = arr.shape
            return scaler.transform(arr.reshape(-1, dd)).reshape(n, s, dd)
        pipeline.X_train_cond = _sc(conds[tr])
        pipeline.X_train_tgt  = _sc(tgts[tr])
        pipeline.X_test_cond  = _sc(conds[te])
        pipeline.X_test_tgt   = _sc(tgts[te])
        pipeline.scaler = scaler
        pipeline.d = d
        print(f"  Simulated CIR: train={len(tr)}, test={len(te)}")
        return pipeline

    else:
        # Real FRED data
        ds_cfg = cfg.get("datasets", {}).get(dataset_name, {})
        start  = ds_cfg.get("start", "2008-01-01")
        end    = ds_cfg.get("end",   "2023-02-16")

        proc_dir = os.path.join("data", "processed")
        proc_prefix = dataset_name

        # Try loading cached processed data
        cond_path = os.path.join(proc_dir, f"{proc_prefix}_train_cond.npy")
        if os.path.exists(cond_path):
            print(f"  Loading cached processed data from {proc_dir}/")
            # return DataPipeline.load(proc_dir, proc_prefix)
            pipeline = DataPipeline.load(proc_dir, proc_prefix)
            pipeline.p = pipeline.X_train_cond.shape[1]
            pipeline.q = pipeline.X_train_tgt.shape[1]
            pipeline.d = pipeline.X_train_cond.shape[-1]
            return pipeline

        # Otherwise, process from raw
        raw_dir = "data/raw"
        levels  = load_fred_csv(raw_dir)
        levels  = filter_date_range(levels, start, end)
        pipeline = DataPipeline(p=p, q=q, train_split=0.8, scaler="standard")
        pipeline.fit_transform(levels)
        pipeline.save(proc_dir, proc_prefix)
        return pipeline


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────
def get_model(model_name: str, d: int, p: int = 10, q: int = 10):
    """Instantiate a model by name."""
    from models.historical.phs    import PHS
    from models.historical.fhs    import FHS
    from models.parametric.garch_model import GARCHModel

    name_map = {
        "phs":      lambda: PHS(p=p, q=q, d=d),
        "fhs":      lambda: FHS(p=p, q=q, d=d),
        "ar":       lambda: GARCHModel(p=p, q=q, d=d, variant="ar"),
        "ar_ret":   lambda: GARCHModel(p=p, q=q, d=d, variant="ar_ret"),
        "garch":    lambda: GARCHModel(p=p, q=q, d=d, variant="garch", dist="normal"),
        "garch_t":  lambda: GARCHModel(p=p, q=q, d=d, variant="garch_t", dist="t"),
        "garch_ret":lambda: GARCHModel(p=p, q=q, d=d, variant="garch",   dist="normal"),
    }

    # NN models — import lazily to avoid hard failures if libraries missing
    def _cgan_fc():
        from models.neural.cgan_fc import CGANFC
        return CGANFC(p=p, q=q, d=d)

    def _cwgan():
        from models.neural.cwgan import CWGAN
        return CWGAN(p=p, q=q, d=d)

    def _cgan_lstm():
        from models.neural.cgan_lstm import CGANLSTM
        return CGANLSTM(p=p, q=q, d=d)

    def _vae():
        from models.neural.vae_model import ConditionalTimeVAE
        return ConditionalTimeVAE(p=p, q=q, d=d)

    name_map.update({
        "cgan_fc":   _cgan_fc,
        "cwgan":     _cwgan,
        "cgan_lstm": _cgan_lstm,
        "vae":       _vae,
    })

    key = model_name.lower()
    if key not in name_map:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(name_map.keys())}")
    return name_map[key]()


# ──────────────────────────────────────────────
# Run single model on single dataset
# ──────────────────────────────────────────────
# def run_model(
#     model,
#     pipeline: DataPipeline,
#     n_synthetic: int = 251,
#     tenors: list = None,
# ) -> dict:
#     """Train, generate, and compute all KPIs for one model × dataset."""
#     d = pipeline.X_train_cond.shape[-1]
#     if tenors is None:
#         tenors = [f"t{j}" for j in range(d)]

#     print(f"\n  → Training {model.name}...")
#     t0 = time.time()
#     model.fit(pipeline.X_train_cond, pipeline.X_train_tgt)
#     train_time = time.time() - t0
#     print(f"     Training done in {train_time:.1f}s")

#     # Generate synthetic paths for all test conditions
#     print(f"  → Generating {n_synthetic} paths per test date...")
#     N_test = pipeline.X_test_cond.shape[0]
#     synth_all = np.zeros((N_test, n_synthetic, pipeline.q, d), dtype=np.float32)

#     for i in range(N_test):
#         synth_all[i] = model.generate(pipeline.X_test_cond[i], n_samples=n_synthetic)

#     # Compute KPIs
#     print(f"  → Computing KPIs...")
#     dist_res = compute_distribution_kpi(pipeline.X_test_tgt, synth_all, tenors)
#     acf_res  = compute_acf_kpi(pipeline.X_test_tgt,  synth_all, tenors=tenors)
#     bt_res   = compute_backtesting_kpi(pipeline.X_test_tgt, synth_all, tenors=tenors)
#     comp     = compute_composite_score(dist_res, acf_res, bt_res)

#     print(f"     DIST={comp['DIST']:.4f}  ACF={comp['ACF']:.4f}  "
#           f"BT={comp['BT']:.4f}  COMP={comp['Composite']:.4f}")

#     return {
#         "scores":    comp,
#         "dist_res":  dist_res,
#         "acf_res":   acf_res,
#         "bt_res":    bt_res,
#         "synth_all": synth_all,
#         "train_time": train_time,
#     }

def run_model(
    model,
    pipeline: DataPipeline,
    n_synthetic: int = 251,
    tenors: list = None,
) -> dict:
    """Train, generate, and compute all KPIs for one model × dataset."""

    d = pipeline.X_train_cond.shape[-1]

    # FIX: get q from target window shape instead of pipeline.q
    q = pipeline.X_test_tgt.shape[1]

    if tenors is None:
        tenors = [f"t{j}" for j in range(d)]

    print(f"\n  → Training {model.name}...")
    t0 = time.time()
    model.fit(pipeline.X_train_cond, pipeline.X_train_tgt)
    train_time = time.time() - t0
    print(f"     Training done in {train_time:.1f}s")

    # Generate synthetic paths for all test conditions
    print(f"  → Generating {n_synthetic} paths per test date...")
    N_test = pipeline.X_test_cond.shape[0]

    # FIXED LINE
    synth_all = np.zeros((N_test, n_synthetic, q, d), dtype=np.float32)

    for i in range(N_test):
        generated = model.generate(
            pipeline.X_test_cond[i],
            n_samples=n_synthetic
        )

        generated = np.asarray(generated, dtype=np.float32)

        # Safety check
        if generated.shape != (n_synthetic, q, d):
            raise ValueError(
                f"{model.name}.generate() returned shape {generated.shape}, "
                f"expected {(n_synthetic, q, d)}"
            )

        synth_all[i] = generated

    # Compute KPIs
    print(f"  → Computing KPIs...")
    dist_res = compute_distribution_kpi(pipeline.X_test_tgt, synth_all, tenors)
    acf_res  = compute_acf_kpi(pipeline.X_test_tgt, synth_all, tenors=tenors)
    bt_res   = compute_backtesting_kpi(pipeline.X_test_tgt, synth_all, tenors=tenors)
    comp     = compute_composite_score(dist_res, acf_res, bt_res)

    print(
        f"     DIST={comp['DIST']:.4f}  ACF={comp['ACF']:.4f}  "
        f"BT={comp['BT']:.4f}  COMP={comp['Composite']:.4f}"
    )

    return {
        "scores": comp,
        "dist_res": dist_res,
        "acf_res": acf_res,
        "bt_res": bt_res,
        "synth_all": synth_all,
        "train_time": train_time,
    }

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
ALL_MODELS   = ["phs", "fhs", "ar_ret", "garch_t", "cgan_fc", "cwgan", "cgan_lstm", "vae"]
ALL_DATASETS = ["usdyc2", "usdyc3", "garch_normal", "garch_t5", "garch_t3", "cir"]
MVP_MODELS   = ["phs", "garch_t", "cwgan", "cgan_fc"]  # Minimum viable


def main():
    parser = argparse.ArgumentParser(description="IQF Project — VaR Model Comparison")
    parser.add_argument("--models",  nargs="+", default=["phs", "garch_t"],
                        help=f"Models to run. 'all' = {ALL_MODELS}. 'mvp' = {MVP_MODELS}")
    parser.add_argument("--dataset", nargs="+", default=["usdyc2"],
                        help=f"Datasets. 'all' = {ALL_DATASETS}")
    parser.add_argument("--n_synth", type=int, default=251, help="Synthetic paths per date")
    parser.add_argument("--out_dir", default="results", help="Results output directory")
    parser.add_argument("--no_plots", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    # Expand 'all' and 'mvp'
    models_to_run = ALL_MODELS if "all" in args.models else \
                    MVP_MODELS  if "mvp" in args.models else args.models
    datasets_to_run = ALL_DATASETS if "all" in args.dataset else args.dataset

    # Load config
    with open("configs/data_config.yaml") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "plots"),  exist_ok=True)

    all_dataset_scores = {}

    for dataset_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f" DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")

        try:
            pipeline = load_dataset(dataset_name, cfg)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        d = pipeline.X_train_cond.shape[-1]
        tenors = ["3m", "6m", "1y", "2y", "3y", "5y", "10y", "20y", "30y"][:d] \
                 if d == 9 else ["3m", "1y"][:d]

        model_scores  = {}
        u_value_store = {}

        for model_name in models_to_run:
            print(f"\n[Model] {model_name.upper()}")
            try:
                model = get_model(model_name, d=d, p=cfg.get("condition_length", 10),
                                  q=cfg.get("target_length", 10))
                results = run_model(model, pipeline, args.n_synth, tenors)
                model_scores[model.name]  = results["scores"]
                # Store u-values for histogram plot
                u_value_store[model.name] = {
                    t: results["bt_res"].get(t, {}).get("u_values", np.array([]))
                    for t in tenors[:2]
                }
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()
                continue

        # Build and print ranking table
        if model_scores:
            df = build_ranking_table(model_scores, dataset_name)
            print_ranking_table(df)

            # Save table
            table_path = os.path.join(args.out_dir, "tables", f"ranking_{dataset_name}.csv")
            df.to_csv(table_path, index=False)
            print(f"Saved: {table_path}")

            all_dataset_scores[dataset_name] = model_scores

            # Plots
            if not args.no_plots:
                u_hist_path = os.path.join(args.out_dir, "plots", f"u_hist_{dataset_name}.png")
                plot_u_value_histograms(
                    u_value_store,
                    tenors_show=tenors[:2],
                    out_path=u_hist_path,
                )

    print("\n" + "="*60)
    print(" EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
