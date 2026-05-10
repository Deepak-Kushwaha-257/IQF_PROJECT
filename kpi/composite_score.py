"""
kpi/composite_score.py
======================
Combine DIST + ACF + BT → Composite Score (Section 4.3.7).

Composite = DIST_score + ACF_score + BT_score
Lower = better.

Also produces model ranking tables matching Tables 14-36 in the paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_composite_score(
    dist_results: dict,
    acf_results: dict,
    bt_results: dict,
) -> dict:
    """
    Combine KPI sub-scores into Composite Score.

    Parameters
    ----------
    dist_results : output of compute_distribution_kpi()
    acf_results  : output of compute_acf_kpi()
    bt_results   : output of compute_backtesting_kpi()

    Returns
    -------
    dict with DIST, ACF, BT, COMP scores
    """
    dist = dist_results.get("__summary__", {})
    acf  = acf_results.get("__summary__", {})
    bt   = bt_results.get("__summary__", {})

    # DIST score = average of Distribution Distance and Series Distance
    dist_score = np.mean([
        dist.get("DIST", 0.5),
        dist.get("SERIES_DIST", 0.5),
    ])

    acf_score = acf.get("ACF", 0.5)
    bt_score  = bt.get("BT",  0.5)
    comp      = dist_score + acf_score + bt_score

    return {
        "DIST":      round(dist_score, 4),
        "ACF":       round(acf_score, 4),
        "BT":        round(bt_score, 4),
        "Composite": round(comp, 4),
    }


def build_ranking_table(
    model_scores: Dict[str, dict],
    dataset_name: str = "Dataset",
) -> pd.DataFrame:
    """
    Build a ranking table like Tables 14-36 in the paper.

    Parameters
    ----------
    model_scores : {model_name: {"DIST":..., "ACF":..., "BT":..., "Composite":...}}
    dataset_name : str

    Returns
    -------
    pd.DataFrame sorted by Composite score ascending (lower = better)
    """
    # Model category mapping
    CATEGORIES = {
        "PHS":       "HS",
        "FHS":       "HS",
        "AR":        "PM",
        "AR-RET":    "PM",
        "GARCH":     "PM",
        "GARCH-RET": "PM",
        "GARCHt-RET":"PM",
        "NS-VS":     "PM",
        "CGAN-FC":   "NN",
        "CGAN-LSTM": "NN",
        "CWGAN":     "NN",
        "DIFFUSION": "NN",
        "SIG":       "NN",
        "VAE":       "NN",
    }

    rows = []
    for model_name, scores in model_scores.items():
        rows.append({
            "Model": model_name,
            "Cat":   CATEGORIES.get(model_name, "?"),
            "DIST":  scores.get("DIST",      None),
            "ACF":   scores.get("ACF",       None),
            "BT":    scores.get("BT",        None),
            "Composite": scores.get("Composite", None),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Composite").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    df.attrs["dataset"] = dataset_name

    return df


def print_ranking_table(df: pd.DataFrame, title: str = None):
    """Pretty-print a ranking table."""
    dataset = df.attrs.get("dataset", "Dataset")
    header  = title or f"Model Comparison — {dataset}"
    print(f"\n{'='*70}")
    print(f" {header}")
    print(f"  KPI scores: lower = better")
    print(f"{'='*70}")
    print(f"  {'Rank':>4}  {'Cat':>4}  {'Model':<12}  "
          f"{'DIST':>8}  {'ACF':>8}  {'BT':>8}  {'Composite':>10}")
    print(f"  {'-'*60}")
    for _, row in df.iterrows():
        print(f"  {int(row['Rank']):>4}  {row['Cat']:>4}  {row['Model']:<12}  "
              f"{row['DIST']:>8.4f}  {row['ACF']:>8.4f}  {row['BT']:>8.4f}  "
              f"{row['Composite']:>10.4f}")
    print(f"{'='*70}\n")


def build_multi_dataset_ranking(
    all_results: Dict[str, Dict[str, dict]],
) -> pd.DataFrame:
    """
    Build a summary ranking table across multiple datasets (like Table 34).

    Parameters
    ----------
    all_results : {dataset_name: {model_name: scores_dict}}

    Returns
    -------
    pd.DataFrame with columns: Model, Cat, dataset1_rank, dataset2_rank, ..., AVG
    """
    datasets = list(all_results.keys())
    models   = list(next(iter(all_results.values())).keys())

    # Rank within each dataset
    rank_data = {ds: {} for ds in datasets}
    for ds, scores in all_results.items():
        sorted_models = sorted(scores.items(), key=lambda x: x[1].get("Composite", 9))
        for rank, (model, _) in enumerate(sorted_models, 1):
            rank_data[ds][model] = rank

    CATEGORIES = {
        "PHS": "HS", "FHS": "HS",
        "AR": "PM", "AR-RET": "PM", "GARCH": "PM",
        "GARCH-RET": "PM", "GARCHt-RET": "PM", "NS-VS": "PM",
        "CGAN-FC": "NN", "CGAN-LSTM": "NN", "CWGAN": "NN",
        "DIFFUSION": "NN", "SIG": "NN", "VAE": "NN",
    }

    rows = []
    for model in models:
        row = {"Model": model, "Cat": CATEGORIES.get(model, "?")}
        ranks = []
        for ds in datasets:
            r = rank_data[ds].get(model, None)
            row[ds] = r
            if r is not None:
                ranks.append(r)
        row["AVG"] = round(np.mean(ranks), 1) if ranks else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("AVG").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df
