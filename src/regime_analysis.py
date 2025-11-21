"""

Detect stress/crash windows from a market index (built from returns),
and compute correlations and volatility during stress vs normal periods and
assesses cluster stability
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
    """Load returns CSV (datetime index, columns = coins)."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def build_market_index(returns: pd.DataFrame) -> pd.Series:
    """
    Build a simple equal-weighted market index (cumulative returns).
    Returns the index level series (float, starting at 1.0).
    """
    eq_ret = returns.mean(axis=1)  # equally weighted daily return
    idx = (1 + eq_ret).cumprod()
    idx.name = "market_index"
    return idx

def compute_drawdown(index: pd.Series) -> pd.Series:
    """Compute drawdown series from an index series."""
    rolling_max = index.cummax()
    drawdown = (index - rolling_max) / rolling_max
    drawdown.name = "drawdown"
    return drawdown


def detect_regimes_by_drawdown(drawdown: pd.Series,
                               crash_thresh: float = -0.25,
                               stress_thresh: float = -0.10,
                               min_len: int = 3) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Identify contiguous windows where drawdown <= thresholds
    """
    regimes = {"crash": [], "stress": []}

    def contiguous_windows(mask):
        #mask is boolean series indexed by dates
        windows = []
        in_window = False
        start = None
        for ts, val in mask.items():
            if val and not in_window:
                in_window = True
                start = ts
            if not val and in_window:
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

    crash_mask = drawdown <= crash_thresh
    stress_mask = (drawdown <= stress_thresh) & (drawdown > crash_thresh)

    regimes["crash"] = contiguous_windows(crash_mask)
    regimes["stress"] = contiguous_windows(stress_mask)
    return regimes


def pick_representative_period(regimes: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
                               kind: str = "stress") -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Pick the longest window of a given kind ('stress' or 'crash').
    Returns (start, end)
    """
    windows = regimes.get(kind, [])
    if not windows:
        raise ValueError(f"No windows found for regime type '{kind}'")
    lengths = [(win[1] - win[0]).days + 1 for win in windows]
    idx = int(np.argmax(lengths))
    return windows[idx]


def slice_returns(returns: pd.DataFrame, window: Tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """Return returns sliced between start and end dates (inclusive)"""
    start, end = window
    return returns.loc[(returns.index >= start) & (returns.index <= end)]


def regime_correlation_and_volatility(returns: pd.DataFrame, window: Tuple[pd.Timestamp, pd.Timestamp]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute correlation matrix and per-coin volatility for the specified window.
    """
    slice_r = slice_returns(returns, window)
    corr = slice_r.corr()
    vol = slice_r.std()
    return corr, vol


def plot_heatmap(matrix: pd.DataFrame, title: str, fname: str):
    """Save heatmap of a square matrix to figures directory."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_volatility_comparison(vol_normal: pd.Series, vol_stress: pd.Series, fname: str):
    """Bar chart comparing volatility normal vs stress per coin"""
    df = pd.DataFrame({"normal_vol": vol_normal, "stress_vol": vol_stress})
    df = df.sort_values("normal_vol", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot.bar(ax=ax)
    ax.set_ylabel("Std of daily returns")
    ax.set_title("Volatility: Normal vs Stress")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_correlation_difference(corr_normal: pd.DataFrame, corr_stress: pd.DataFrame, fname: str):
    """Plot heatmap of (stress - normal) correlation differences"""
    diff = corr_stress - corr_normal
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(diff, annot=True, fmt=".2f", cmap="bwr", center=0, ax=ax)
    ax.set_title("Correlation Difference (Stress − Normal)")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def cluster_stability(returns: pd.DataFrame,
                      normal_window: Tuple[pd.Timestamp, pd.Timestamp],
                      stress_window: Tuple[pd.Timestamp, pd.Timestamp],
                      k: int = 4) -> pd.DataFrame:
    """
    Run KMeans on normal and stress periods and return a contingency table
    showing how coins switch clusters.
    """
    r_normal = slice_returns(returns, normal_window)
    r_stress = slice_returns(returns, stress_window)
    # features: correlation rows averaged over period
    corr_normal = r_normal.corr()
    corr_stress = r_stress.corr()

    # use correlation rows as features (coins x coins)
    Xn = corr_normal.values
    Xs = corr_stress.values

    km_n = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xn)
    km_s = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xs)

    df = pd.DataFrame({
        "coin": corr_normal.index,
        "cluster_normal": km_n,
        "cluster_stress": km_s
    })
    # contingency matrix
    cont = pd.crosstab(df["cluster_normal"], df["cluster_stress"])
    return df, cont


def save_csv(obj, fname: str):
    """Save DataFrame/Series"""
    path = os.path.join(OUTPUTS_DIR, fname)
    if isinstance(obj, pd.Series):
        obj.to_csv(path)
    else:
        obj.to_csv(path)
    print(f"Saved: {path}")


def main():
    returns = load_returns(RETURNS_PATH)

    # build market index and drawdown
    market_idx = build_market_index(returns)
    save_csv(market_idx, "market_index.csv")
    drawdown = compute_drawdown(market_idx)
    save_csv(drawdown, "market_drawdown.csv")

    # detect regimes
    regimes = detect_regimes_by_drawdown(drawdown, crash_thresh=-0.25, stress_thresh=-0.10, min_len=3)
    #save regimes (start/end lists)
    regimes_df = []
    for kind in ["crash", "stress"]:
        for start, end in regimes[kind]:
            regimes_df.append({"kind": kind, "start": start, "end": end})
    if regimes_df:
        save_csv(pd.DataFrame(regimes_df), "detected_regimes.csv")
    else:
        print("No regimes detected with current thresholds; consider adjusting thresholds.")

    # pick representative windows
    try:
        stress_window = pick_representative_period(regimes, "stress")
    except ValueError:
        # fallback: use the worst drawdown segment if no stress windows found
        worst_date = drawdown.idxmin()
        stress_window = (worst_date, worst_date)
        print("No stress window found; using single-day worst drawdown as stress window.")

    # normal window: pick a period with small drawdown (near zero) of same length
    stress_len = (stress_window[1] - stress_window[0]).days + 1
    # find center date for normal period: pick the date with drawdown closest to zero and enough room around
    valid_dates = drawdown.index
    candidate = None
    for center in valid_dates:
        start = center - pd.Timedelta(days=stress_len // 2)
        end = start + pd.Timedelta(days=stress_len - 1)
        if start >= valid_dates[0] and end <= valid_dates[-1]:
            window_dd = drawdown.loc[start:end].abs().max()
            if window_dd < 0.02:
                candidate = (start, end)
                break
    if candidate is None:
        # fallback: take earliest available window of same length
        candidate = (valid_dates[0], valid_dates[0] + pd.Timedelta(days=stress_len - 1))
    normal_window = candidate

    # compute correlations and volatilities
    corr_stress, vol_stress = regime_correlation_and_volatility(returns, stress_window)
    corr_normal, vol_normal = regime_correlation_and_volatility(returns, normal_window)

    # save matrices and vols
    save_csv(corr_stress, "corr_stress.csv")
    save_csv(corr_normal, "corr_normal.csv")
    save_csv(vol_stress, "vol_stress.csv")
    save_csv(vol_normal, "vol_normal.csv")

    # plots: heatmaps, volatility comparison, difference map
    plot_heatmap(corr_normal, "Correlation - Normal Period", "corr_normal_heatmap.png")
    plot_heatmap(corr_stress, "Correlation - Stress Period", "corr_stress_heatmap.png")
    plot_correlation_difference(corr_normal, corr_stress, "corr_diff_heatmap.png")
    plot_volatility_comparison(vol_normal, vol_stress, "volatility_comparison.png")

    # cluster stability
    df_clusters, contingency = cluster_stability(returns, normal_window, stress_window, k=4)
    save_csv(df_clusters, "cluster_labels_normal_vs_stress.csv")
    save_csv(contingency, "cluster_contingency_table.csv")

    # plot contingency as heatmap
    plot_heatmap(contingency, "Cluster Contingency (normal vs stress)", "cluster_contingency.png")

    print("Regime analysis complete. Outputs and figures saved.")


if __name__ == "__main__":
    main()
