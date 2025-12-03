"""
Compute and visualize rolling and EWMA correlations for major cryptocurrencies.

This script loads daily returns, calculates 90-day rolling correlations
and 60-day EWMA correlations, then generates plots for key coin pairs
and the overall mean correlation across all coins.
"""

import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "results/data/returns.csv"
OUTPUTS_FOLDER = "results/outputs"
FIGURES_FOLDER = "results/figures"

os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)

ROLLING_WINDOW = 90
EWMA_SPAN = 60
KEY_PAIRS = [("bitcoin", "ethereum"), ("bitcoin", "solana"), ("ethereum", "solana")]


def load_returns(file_path: str) -> pd.DataFrame:
    """Load daily returns into a DataFrame with datetime index."""
    return pd.read_csv(file_path, index_col=0, parse_dates=True)


def rolling_correlation(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling correlation matrix, output one row per timestamp, coins as columns"""
    # Rolling correlation produces multi-index
    corr = df.rolling(window).corr()
    # Take mean correlation of each coin with all coins
    corr_mean = corr.groupby(level=0).mean()
    # Forward-fill missing values
    corr_mean = corr_mean.ffill().bfill()
    return corr_mean


def ewma_correlation(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """EWMA correlation, one row per timestamp, coins as columns"""
    corr = df.ewm(span=span).corr()
    corr_mean = corr.groupby(level=0).mean()
    corr_mean = corr_mean.ffill().bfill()
    return corr_mean


def plot_selected_pairs(corr_df: pd.DataFrame, pairs: List[tuple], figures_folder: str, title: str):
    """Plot correlation for selected key coin pairs"""
    plt.figure(figsize=(12, 6))
    for c1, c2 in pairs:
        col_name = f"{c1}-{c2}"
        if col_name in corr_df.columns:
            plt.plot(corr_df.index, corr_df[col_name].values, label=f"{c1.upper()}–{c2.upper()}")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    # save plot to figures folder
    fname = title.lower().replace(" ", "_") + "_pairs.png"
    plt.savefig(os.path.join(figures_folder, fname))
    plt.close()


def plot_average_correlation(corr_df: pd.DataFrame, figures_folder: str, title: str):
    """Plot mean correlation across all coins to identify correlation regimes"""
    plt.figure(figsize=(12, 6))
    plt.plot(corr_df.index, corr_df.mean(axis=1), color="navy")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Mean Correlation")
    plt.tight_layout()
    # save plot and CSV
    fname = title.lower().replace(" ", "_") + "_mean.png"
    plt.savefig(os.path.join(figures_folder, fname))
    corr_df.mean(axis=1).to_csv(os.path.join(OUTPUTS_FOLDER, fname.replace(".png", ".csv")))
    plt.close()


def detect_correlation_regimes(corr_df: pd.DataFrame, title: str, threshold_std: float = 1.0) -> pd.DataFrame:
    """
    Identify high/low correlation regimes.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs("results/outputs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # Compute mean correlation 
    mean_corr = corr_df.mean(axis=1)
    mean_val = mean_corr.mean()
    std_val = mean_corr.std()
    high_thresh = mean_val + threshold_std * std_val
    low_thresh = mean_val - threshold_std * std_val

    regimes = pd.Series(index=mean_corr.index, dtype="object")
    regimes[mean_corr >= high_thresh] = "high"
    regimes[mean_corr <= low_thresh] = "low"
    regimes[(mean_corr < high_thresh) & (mean_corr > low_thresh)] = "normal"

    # Save
    regimes_df = pd.DataFrame({"mean_correlation": mean_corr, "regime": regimes})
    csv_path = os.path.join("results/outputs", f"{title.lower().replace(' ','_')}_regimes.csv")
    regimes_df.to_csv(csv_path)

    # Plot shaded regions
    plt.figure(figsize=(12,6))
    plt.plot(mean_corr.index, mean_corr.values, color="navy", label="Mean Correlation")
    for i, label in regimes.items():
        if label == "high":
            plt.axvspan(i, i, color="red", alpha=0.1)
        elif label == "low":
            plt.axvspan(i, i, color="green", alpha=0.1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Mean Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join("results/figures", f"{title.lower().replace(' ','_')}_shaded.png"))
    plt.close()

    return regimes_df


def main():
    """Load returns, compute rolling/EWMA correlations, and plot results"""
    returns = load_returns(DATA_FILE)

    # Rolling correlations
    rolling_corr = rolling_correlation(returns, ROLLING_WINDOW)
    rolling_corr.to_csv(os.path.join(OUTPUTS_FOLDER, f"rolling_corr_{ROLLING_WINDOW}d.csv"))
    plot_selected_pairs(rolling_corr, KEY_PAIRS, FIGURES_FOLDER, f"Rolling {ROLLING_WINDOW}-Day")
    plot_average_correlation(rolling_corr, FIGURES_FOLDER, f"Rolling {ROLLING_WINDOW}-Day")

    # EWMA correlations
    ewma_corr = ewma_correlation(returns, EWMA_SPAN)
    ewma_corr.to_csv(os.path.join(OUTPUTS_FOLDER, f"ewma_corr_{EWMA_SPAN}d.csv"))
    plot_selected_pairs(ewma_corr, KEY_PAIRS, FIGURES_FOLDER, f"EWMA {EWMA_SPAN}-Day")
    plot_average_correlation(ewma_corr, FIGURES_FOLDER, f"EWMA {EWMA_SPAN}-Day")
    
    # detect regimes
    detect_correlation_regimes(rolling_corr, f"Rolling {ROLLING_WINDOW}-Day Regimes")
    detect_correlation_regimes(ewma_corr, f"EWMA {EWMA_SPAN}-Day Regimes")


if __name__ == "__main__":
    main()
