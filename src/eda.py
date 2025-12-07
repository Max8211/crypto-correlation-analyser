"""
Analyse returns by computing
summary statistics, visualizing distributions, correlations, and rolling
volatility. 
Extreme outliers are clipped for visualization
"""

import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "results/data/returns.csv"
OUTPUTS_FOLDER = "results/outputs"
FIGURES_FOLDER = "results/figures"

os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)


def load_returns(file_path: str) -> pd.DataFrame:
    """Load the returns dataset from CSV."""
    return pd.read_csv(file_path, index_col=0, parse_dates=True)


def summary_statistics(df: pd.DataFrame, output_path: str) -> None:
    """Compute and save basic descriptive statistics."""
    stats = df.describe().T
    stats["skew"] = df.skew()
    stats["kurtosis"] = df.kurtosis()
    stats.to_csv(output_path)


def correlation_heatmap(df: pd.DataFrame, figures_folder: str, output_path: str) -> None:
    """Compute correlation matrix and plot heatmap"""
    corr = df.corr()
    corr.to_csv(output_path)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap of Returns")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "correlation_heatmap.png"))
    plt.close()


def plot_rolling_volatility(df: pd.DataFrame, figures_folder: str, window: int = 30) -> None:
    """Plot rolling volatility with clipping and log scale for readability."""
    rolling_vol = df.rolling(window=window).std()
    # Clip extremes for visualization
    lower, upper = rolling_vol.quantile(0.01).min(), rolling_vol.quantile(0.99).max()
    rolling_vol_clip = rolling_vol.clip(lower, upper, axis=1)

    plt.figure(figsize=(12, 6))
    for coin in df.columns:
        plt.plot(rolling_vol_clip.index, rolling_vol_clip[coin], label=coin)

    plt.title(f"{window}-Day Rolling Volatility (Clipped 1-99%, Log Scale)")
    plt.ylabel("Volatility")
    plt.xlabel("Date")
    plt.yscale("log")  # log scale
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, f"rolling_volatility_{window}d_log.png"))
    plt.close()


def main():
    df = load_returns(DATA_FILE)

    # Save 
    summary_statistics(df, os.path.join(OUTPUTS_FOLDER, "summary_statistics.csv"))

    # Compute and plot correlation heatmap
    correlation_heatmap(
        df, FIGURES_FOLDER, os.path.join(OUTPUTS_FOLDER, "correlation_matrix.csv")
    )

    # Plot rolling volatility
    plot_rolling_volatility(df, FIGURES_FOLDER, window=30)

if __name__ == "__main__":
    main()