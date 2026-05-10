"""
utils/visualization.py
=======================
All visualization functions matching the paper's figures.

Figures reproduced:
  - Figure 6:  Empirical distribution histograms + ACF plots
  - Figure 7:  PCA visualization
  - Figure 8:  t-SNE visualization
  - Figure 9:  UMAP visualization
  - Figure 10: Sample mean histograms
  - Figure 11: Inter-tenor correlation heatmap
  - Figure 12: Correlation matrix distance
  - Figure 16: u-value histograms
  - Figure 17/18: Envelope plots (5th/95th quantile vs realized)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# Optional dimensionality reduction
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

TENORS_9 = ["3m", "6m", "1y", "2y", "3y", "5y", "10y", "20y", "30y"]
TENORS_2 = ["3m", "1y"]


# ──────────────────────────────────────────────
# Figure 6: Histogram + ACF comparison
# ──────────────────────────────────────────────
def plot_distribution_acf(
    X_real: np.ndarray,
    X_synth_avg: np.ndarray,
    model_name: str = "Model",
    tenors: list = None,
    n_show: int = 3,
    out_path: str = None,
):
    """
    Plot empirical distribution histograms (raw + log scale) and ACF.
    Matches Figure 6 in paper.

    Parameters
    ----------
    X_real      : (N, q, d)
    X_synth_avg : (N, q, d) — mean synthetic path per test date
    n_show      : number of tenors to show
    """
    if tenors is None:
        tenors = TENORS_9[:X_real.shape[-1]]
    d_show = min(n_show, X_real.shape[-1])
    show_tenors = tenors[:d_show]

    fig, axes = plt.subplots(d_show, 3, figsize=(14, 3 * d_show))
    if d_show == 1:
        axes = axes[np.newaxis]

    fig.suptitle(f"Empirical Distribution & ACF — {model_name}", fontsize=11)

    for i, tenor in enumerate(show_tenors):
        j = i  # tenor index
        real_flat  = X_real[:, :, j].flatten()
        synth_flat = X_synth_avg[:, :, j].flatten()

        # Raw histogram
        ax = axes[i, 0]
        ax.hist(real_flat,  bins=60, alpha=0.6, label="Historical", density=True, color="steelblue")
        ax.hist(synth_flat, bins=60, alpha=0.6, label="Generated",  density=True, color="darkorange")
        ax.set_title(f"Raw scale: {tenor}")
        ax.legend(fontsize=7)

        # Log-scale histogram
        ax = axes[i, 1]
        bins = np.linspace(min(real_flat.min(), synth_flat.min()),
                           max(real_flat.max(), synth_flat.max()), 80)
        ax.hist(real_flat,  bins=bins, alpha=0.6, label="Historical", density=True,
                color="steelblue", log=True)
        ax.hist(synth_flat, bins=bins, alpha=0.6, label="Generated",  density=True,
                color="darkorange", log=True)
        ax.set_title(f"Log scale: {tenor}")
        ax.legend(fontsize=7)

        # ACF plot (lag 1..20)
        ax = axes[i, 2]
        max_lag = 20
        real_series  = X_real[:, :, j].flatten()
        synth_series = X_synth_avg[:, :, j].flatten()
        acf_real  = [np.corrcoef(real_series[:-k],  real_series[k:])[0, 1]  for k in range(1, max_lag+1)]
        acf_synth = [np.corrcoef(synth_series[:-k], synth_series[k:])[0, 1] for k in range(1, max_lag+1)]
        lags = range(1, max_lag + 1)
        ax.plot(lags, acf_real,  label="Historical", color="steelblue")
        ax.plot(lags, acf_synth, label="Generated",  color="darkorange")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"ACF: {tenor}")
        ax.set_ylim(-0.3, 1.0)
        ax.legend(fontsize=7)

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.show()
    return fig


# ──────────────────────────────────────────────
# Figure 11: Inter-tenor correlation heatmap
# ──────────────────────────────────────────────
def plot_correlation_matrix(
    X: np.ndarray,
    title: str = "Correlation Matrix",
    tenors: list = None,
    out_path: str = None,
):
    returns_flat = X.reshape(-1, X.shape[-1])
    corr = np.corrcoef(returns_flat.T)
    if tenors is None:
        tenors = TENORS_9[:X.shape[-1]]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=tenors, yticklabels=tenors,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return fig


# ──────────────────────────────────────────────
# Figure 16: u-value histograms
# ──────────────────────────────────────────────
def plot_u_value_histograms(
    u_value_dict: dict,
    tenors_show: list = None,
    out_path: str = None,
):
    """
    u_value_dict : {model_name: {tenor: u_values_array}}
    """
    models = list(u_value_dict.keys())
    if tenors_show is None:
        tenors_show = ["3m", "10y"]

    n_tenors = len(tenors_show)
    fig, axes = plt.subplots(1, n_tenors, figsize=(6 * n_tenors, 4))
    if n_tenors == 1:
        axes = [axes]

    fig.suptitle("Histogram of u-values (1-day returns)", fontsize=11)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    bins = np.linspace(0, 1, 11)  # 10 bins
    width = 0.8 / len(models)

    for t_idx, tenor in enumerate(tenors_show):
        ax = axes[t_idx]
        for m_idx, model_name in enumerate(models):
            u_vals = u_value_dict[model_name].get(tenor, None)
            if u_vals is None:
                continue
            counts, _ = np.histogram(u_vals, bins=bins)
            n = len(u_vals)
            heights = (counts / n) * 10   # normalize so uniform = 1.0
            x_pos = bins[:-1] + 0.05 + m_idx * width
            ax.bar(x_pos, heights, width=width * 0.9, color=colors[m_idx],
                   label=model_name, alpha=0.85)

        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="Uniform")
        ax.set_title(f"u-values: {tenor}")
        ax.set_xlabel("u-value")
        ax.set_ylabel("Normalized frequency")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.show()
    return fig


# ──────────────────────────────────────────────
# Figure 17/18: Envelope plots
# ──────────────────────────────────────────────
def plot_envelope(
    dates: np.ndarray,
    realized: np.ndarray,
    q05: np.ndarray,
    q95: np.ndarray,
    model_name: str = "Model",
    tenor: str = "3m",
    breach_info: str = "",
    out_path: str = None,
):
    """
    Envelope plot: realized return vs 5th/95th quantile of forecast distribution.
    Matches Figures 17-18 in paper.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, realized, color="black", linewidth=0.6, label=tenor, alpha=0.8)
    ax.plot(dates, q05, color="darkorange", linewidth=0.8, linestyle="--", label="q05")
    ax.plot(dates, q95, color="darkorange", linewidth=0.8, linestyle="--", label="q95")
    ax.fill_between(dates, q05, q95, alpha=0.2, color="darkorange")

    ax.axhline(0, color="gray", linewidth=0.4)
    ax.set_title(f"Realized and quantile plot — {tenor}, {model_name}\n{breach_info}")
    ax.legend(fontsize=8)
    ax.set_ylabel("Return")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.show()
    return fig


# ──────────────────────────────────────────────
# PCA visualization
# ──────────────────────────────────────────────
def plot_pca(
    X_real: np.ndarray,
    X_synth_flat: np.ndarray,
    model_name: str = "Model",
    out_path: str = None,
):
    if not PCA_AVAILABLE:
        print("scikit-learn not installed for PCA")
        return None

    N_r = min(len(X_real), 1000)
    N_s = min(len(X_synth_flat), 1000)
    Xr = X_real[:N_r].reshape(N_r, -1)
    Xs = X_synth_flat[:N_s].reshape(N_s, -1)

    pca = PCA(n_components=2)
    combined = np.vstack([Xr, Xs])
    pc = pca.fit_transform(combined)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"PCA — {model_name}", fontsize=11)

    # Combined
    axes[0].scatter(pc[:N_r, 0], pc[:N_r, 1], s=5, c="green",  label="real",      alpha=0.5)
    axes[0].scatter(pc[N_r:, 0], pc[N_r:, 1], s=5, c="blue",   label="synthetic", alpha=0.5)
    axes[0].set_title("PCA (real + synthetic)")
    axes[0].legend(fontsize=8)

    # Real only
    axes[1].scatter(pc[:N_r, 0], pc[:N_r, 1], s=8, c="green", alpha=0.6)
    axes[1].set_title("PCA (real only)")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return fig
