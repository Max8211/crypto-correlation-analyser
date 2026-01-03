"""
Detect stress windows using Downside Volatility.
Aggregate Method: Computes average stats for all Normal vs Stress days.
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RETURNS_PATH = "results/data/returns.csv"
OUTPUTS_DIR = "results/outputs"
FIGURES_DIR = "results/figures"

# volatility parameters
VOL_WINDOW = 14
# Top 15% of downside volatility days (to absorb most historical crises)
VOL_PERCENTILE = 0.85

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_returns(path: str) -> pd.DataFrame:
    """Load dataset ensuring the index is treated as datetime objects."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def build_market_index(returns: pd.DataFrame) -> pd.Series:
    """
    Construct an equally-weighted market index
    Uses cumulative product of mean returns to track market growth
    """
    eq_ret = returns.mean(axis=1)
    idx = (1 + eq_ret).cumprod()
    idx.name = "market_index"
    return idx


def compute_downside_volatility(returns: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate risk based only on negative price movements
    removes positive returns to focus on 'bad' volatility (risk)
    """
    market_ret = returns.mean(axis=1)
    neg_ret = market_ret.copy()
    neg_ret[neg_ret > 0] = 0

    downside_vol = neg_ret.rolling(window=window).std()
    downside_vol.name = "downside_volatility"
    return downside_vol


# Identify continuous periods of stress. use a 2-day minimum length to filter out one-day outliers
def detect_stress_periods(
    stress_mask: pd.Series, min_len: int = 2
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Identify continuous periods where stress_mask is True.
    """
    windows = []
    in_window = False
    start = None
    prev_ts = None

    clean_series = stress_mask.dropna()

    for ts, is_stress in clean_series.items():
        if is_stress and not in_window:
            in_window = True
            start = ts
        elif not is_stress and in_window:
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


def plot_heatmap(matrix: pd.DataFrame, title: str, fname: str):
    """
    Standard Heatmap. No dendrograms. No annotations.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="coolwarm", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    returns = load_returns(RETURNS_PATH)

    # generate and save market index
    market_idx = build_market_index(returns)
    market_idx.to_csv(os.path.join(OUTPUTS_DIR, "market_index.csv"))

    # Compute downside vol
    rolling_downside = compute_downside_volatility(returns, window=VOL_WINDOW)
    rolling_downside.to_csv(os.path.join(OUTPUTS_DIR, "market_downside_vol.csv"))

    # Determine Threshold (Expanding Window to ensure it evolves with the market)
    dynamic_threshold = rolling_downside.expanding(min_periods=365).quantile(
        VOL_PERCENTILE
    )

    # Create Boolean Mask
    is_stress_mask = rolling_downside > dynamic_threshold
    print(f"Computed dynamic volatility thresholds (Expanding Window).")

    # Detect Stress Periods
    stress_windows = detect_stress_periods(stress_mask=is_stress_mask, min_len=2)

    # Save Regimes to CSV
    regimes_df = []
    for start, end in stress_windows:
        regimes_df.append({"kind": "stress", "start": start, "end": end})

    # fill the gaps between stress windows with 'normal' labels
    all_dates = rolling_downside.dropna().index
    stress_intervals = [(s, e) for s, e in stress_windows]

    if stress_intervals:
        # Initialize the tracker at the start of the dataset
        prev_end = all_dates[0] - pd.Timedelta(days=1)
        for s, e in stress_intervals:
            # Check if there is a gap (Normal period) before this Stress window starts
            if (s - prev_end).days > 1:
                normal_start = prev_end + pd.Timedelta(days=1)
                normal_end = s - pd.Timedelta(days=1)
                # label as normal if gap is valid
                if normal_start <= normal_end:
                    regimes_df.append(
                        {"kind": "normal", "start": normal_start, "end": normal_end}
                    )
            prev_end = e
        if prev_end < all_dates[-1]:
            regimes_df.append(
                {
                    "kind": "normal",
                    "start": prev_end + pd.Timedelta(days=1),
                    "end": all_dates[-1],
                }
            )
    else:
        # If No stress was ever detected, the entire dataset is labeled Normal
        if not all_dates.empty:
            regimes_df.append(
                {"kind": "normal", "start": all_dates[0], "end": all_dates[-1]}
            )

    if regimes_df:
        pd.DataFrame(regimes_df).to_csv(
            os.path.join(OUTPUTS_DIR, "detected_regimes.csv"), index=False
        )
        print(f"Detected {len(stress_windows)} stress windows.")

    if not stress_windows:
        print("No stress windows detected. Exiting.")
        return

    #  Statistical Aggregate Approach
    valid_index = rolling_downside.dropna().index
    # Final alignment: only use dates where we have volatility and return data
    final_stress_mask = is_stress_mask.loc[valid_index]
    final_normal_mask = ~final_stress_mask
    returns_valid = returns.loc[valid_index]

    # Compute correlations for EVERY day in each regime
    corr_normal = returns_valid[final_normal_mask].corr()
    corr_stress = returns_valid[final_stress_mask].corr()

    # Compute average volatility for EVERY day in each regime
    vol_normal = returns_valid[final_normal_mask].std()
    vol_stress = returns_valid[final_stress_mask].std()

    print(
        f"Aggregate Stats: {final_normal_mask.sum()} Normal days, {final_stress_mask.sum()} Stress days."
    )

    # Save Volatilities
    vol_stress.to_csv(os.path.join(OUTPUTS_DIR, "vol_stress.csv"))
    vol_normal.to_csv(os.path.join(OUTPUTS_DIR, "vol_normal.csv"))

    # Save Correlations
    corr_stress.to_csv(os.path.join(OUTPUTS_DIR, "corr_stress.csv"))
    corr_normal.to_csv(os.path.join(OUTPUTS_DIR, "corr_normal.csv"))

    # Standard Heatmaps
    plot_heatmap(
        corr_normal, "Correlation - Normal (Aggregate)", "corr_normal_heatmap.png"
    )
    plot_heatmap(
        corr_stress, "Correlation - Stress (Aggregate)", "corr_stress_heatmap.png"
    )

    # Diff Heatmap
    corr_diff = corr_stress - corr_normal
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_diff, annot=False, cmap="coolwarm", center=0, vmin=-0.5, vmax=0.5, ax=ax
    )
    ax.set_title("Correlation Difference (Stress - Normal)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURES_DIR, "corr_diff_heatmap.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    print("Analysis complete.")


def load_regime_outputs():
    """
    Load precomputed regime outputs for main.py metrics
    """
    OUTPUTS_DIR = "results/outputs"

    corr_normal = pd.read_csv(os.path.join(OUTPUTS_DIR, "corr_normal.csv"), index_col=0)
    corr_stress = pd.read_csv(os.path.join(OUTPUTS_DIR, "corr_stress.csv"), index_col=0)

    # Load volatility stats (ensure they are Series, not 1-column DataFrames)
    vol_normal = pd.read_csv(os.path.join(OUTPUTS_DIR, "vol_normal.csv"), index_col=0)
    if vol_normal.shape[1] == 1:
        vol_normal = vol_normal.iloc[:, 0]

    vol_stress = pd.read_csv(os.path.join(OUTPUTS_DIR, "vol_stress.csv"), index_col=0)
    if vol_stress.shape[1] == 1:
        vol_stress = vol_stress.iloc[:, 0]

    # Load cluster labels if they exist
    cluster_labels_path = os.path.join(OUTPUTS_DIR, "cluster_labels.csv")
    if os.path.exists(cluster_labels_path):
        cluster_labels = pd.read_csv(cluster_labels_path, index_col=0).iloc[:, 0]
    else:
        cluster_labels = pd.Series(dtype=float)

    return corr_normal, corr_stress, vol_normal, vol_stress, cluster_labels


if __name__ == "__main__":
    main()
