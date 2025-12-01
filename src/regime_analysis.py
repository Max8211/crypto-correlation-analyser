"""
Detect stress windows from a market index (built from returns),
and compute correlations and volatility during stress vs normal periods.
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

RETURNS_PATH = "results/data/returns.csv"
OUTPUTS_DIR = "results/outputs"
FIGURES_DIR = "results/figures"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_returns(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)

def build_market_index(returns: pd.DataFrame) -> pd.Series:
    eq_ret = returns.mean(axis=1)
    idx = (1 + eq_ret).cumprod()
    idx.name = "market_index"
    return idx

def compute_drawdown(index: pd.Series) -> pd.Series:
    rolling_max = index.cummax()
    drawdown = (index - rolling_max) / rolling_max
    drawdown.name = "drawdown"
    return drawdown

def detect_stress_periods(drawdown: pd.Series,
                          stress_thresh: float = -0.10,
                          min_len: int = 3) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Detect contiguous windows where drawdown exceeds the stress threshold.
    Returns a list of (start, end) tuples.
    """
    windows = []
    in_window = False
    start = None
    for ts, val in drawdown.items():
        if val <= stress_thresh and not in_window:
            in_window = True
            start = ts
        elif val > stress_thresh and in_window:
            end = prev_ts
            if (end - start).days + 1 >= min_len:
                windows.append((start, end))
            in_window = False
        prev_ts = ts
    if in_window:
        end = prev_ts
        if (end - start).days + 1 >= min_len:
            windows.append((start, end))
    return windows

def pick_representative_period(windows: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if not windows:
        raise ValueError("No stress periods found")
    lengths = [(win[1] - win[0]).days + 1 for win in windows]
    idx = int(np.argmax(lengths))
    return windows[idx]

def slice_returns(returns: pd.DataFrame, window: Tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    start, end = window
    return returns.loc[(returns.index >= start) & (returns.index <= end)]

def regime_correlation_and_volatility(returns: pd.DataFrame, window: Tuple[pd.Timestamp, pd.Timestamp]):
    slice_r = slice_returns(returns, window)
    corr = slice_r.corr()
    vol = slice_r.std()
    return corr, vol

def plot_heatmap(matrix: pd.DataFrame, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=0.2, vmax=1.0, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def main():
    returns = load_returns(RETURNS_PATH)

    # Index + drawdown
    market_idx = build_market_index(returns)
    market_idx.to_csv(os.path.join(OUTPUTS_DIR, "market_index.csv"))

    drawdown = compute_drawdown(market_idx)
    drawdown.to_csv(os.path.join(OUTPUTS_DIR, "market_drawdown.csv"))

    # Detect stress periods
    stress_windows = detect_stress_periods(drawdown, stress_thresh=-0.10)

    # Save detected stress periods
    regimes_df = []
    for start, end in stress_windows:
        regimes_df.append({"kind": "stress", "start": start, "end": end})
    if regimes_df:
        pd.DataFrame(regimes_df).to_csv(os.path.join(OUTPUTS_DIR, "detected_regimes.csv"))

    # Pick representative stress window
    stress_window = pick_representative_period(stress_windows)
    stress_len = (stress_window[1] - stress_window[0]).days + 1

    # Normal window: period with low drawdown, same length as stress
    valid_dates = drawdown.index
    candidate = None
    for center in valid_dates:
        start = center - pd.Timedelta(days=stress_len // 2)
        end = start + pd.Timedelta(days=stress_len - 1)
        if start >= valid_dates[0] and end <= valid_dates[-1]:
            if drawdown.loc[start:end].abs().max() < 0.02:
                candidate = (start, end)
                break
    if candidate is None:
        candidate = (valid_dates[0], valid_dates[0] + pd.Timedelta(days=stress_len - 1))
    normal_window = candidate

    # Compute correlations and volatility
    corr_stress, vol_stress = regime_correlation_and_volatility(returns, stress_window)
    corr_normal, vol_normal = regime_correlation_and_volatility(returns, normal_window)

    corr_stress.to_csv(os.path.join(OUTPUTS_DIR, "corr_stress.csv"))
    corr_normal.to_csv(os.path.join(OUTPUTS_DIR, "corr_normal.csv"))
    vol_stress.to_csv(os.path.join(OUTPUTS_DIR, "vol_stress.csv"))
    vol_normal.to_csv(os.path.join(OUTPUTS_DIR, "vol_normal.csv"))

    # Plot heatmaps
    plot_heatmap(corr_normal, "Correlation - Normal Period", "corr_normal_heatmap.png")
    plot_heatmap(corr_stress, "Correlation - Stress Period", "corr_stress_heatmap.png")
    corr_diff = corr_stress - corr_normal
    # Difference heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_diff, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            vmin=-0.5, vmax=0.5, ax=ax)  # <-- adjust these limits
    ax.set_title("Correlation Difference (Stress - Normal)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "corr_diff_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(FIGURES_DIR, 'corr_diff_heatmap.png')}")
    # Clustering stability
    r_normal = slice_returns(returns, normal_window)
    r_stress = slice_returns(returns, stress_window)
    _ = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(r_normal.corr().values)
    _ = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(r_stress.corr().values)

    print("Regime analysis complete. Outputs and figures saved.")

if __name__ == "__main__":
    main()

def load_regime_outputs():
    """
    Load precomputed regime outputs for main.py metrics
    Returns:
        corr_normal : pd.DataFrame
        corr_stress : pd.DataFrame
        vol_normal  : pd.Series
        vol_stress  : pd.Series
        cluster_labels : pd.Series
    """
    OUTPUTS_DIR = "results/outputs"

    # Correlations
    corr_normal = pd.read_csv(f"{OUTPUTS_DIR}/corr_normal.csv", index_col=0)
    corr_stress  = pd.read_csv(f"{OUTPUTS_DIR}/corr_stress.csv", index_col=0)

    # Volatility
    vol_normal = pd.read_csv(f"{OUTPUTS_DIR}/vol_normal.csv", index_col=0)
    if vol_normal.shape[1] == 1:
        vol_normal = vol_normal.iloc[:, 0]

    vol_stress = pd.read_csv(f"{OUTPUTS_DIR}/vol_stress.csv", index_col=0)
    if vol_stress.shape[1] == 1:
        vol_stress = vol_stress.iloc[:, 0]

    # Cluster labels fallback
    cluster_labels_path = f"{OUTPUTS_DIR}/cluster_labels.csv"
    if os.path.exists(cluster_labels_path):
        cluster_labels = pd.read_csv(cluster_labels_path, index_col=0, squeeze=True)
    else:
        cluster_labels = pd.Series([0, 1, 2, 3], index=[0,1,2,3])

    return corr_normal, corr_stress, vol_normal, vol_stress, cluster_labels