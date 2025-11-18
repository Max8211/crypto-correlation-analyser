"""
Analyse returns by computing
summary statistics, visualizing distributions, correlations, and rolling
volatility. 

Extreme outliers are clipped for visualization to make comparisons readable.
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


def plot_distributions(df: pd.DataFrame, figures_folder: str) -> None:
    """Generate histograms for each coin with shared x-axis"""
    # Compute 1st and 99th percentiles across all coins for x-axis limits
    lower, upper = df.quantile(0.01).min(), df.quantile(0.99).max()

    for coin in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[coin].clip(lower, upper), bins=50, kde=True, color="skyblue")
        plt.title(f"{coin.capitalize()} Return Distribution")
        plt.xlabel("Daily Return")
        plt.ylabel("Frequency")
        plt.xlim(lower, upper)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, f"{coin}_distribution.png"))
        plt.close()


def plot_boxplots(df: pd.DataFrame, figures_folder: str) -> None:
    """Generate a boxplot for all coins with clipped extremes"""
    # axis=1 aligns Series of min/max to columns
    df_clip = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_clip, palette="Set2")
    plt.title("Daily Returns Boxplot (Clipped 1-99%)")
    plt.ylabel("Daily Return")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "boxplot_returns.png"))
    plt.close()


def correlation_heatmap(df: pd.DataFrame, figures_folder: str, output_path: str) -> None:
    """Compute correlation matrix and plot heatmap."""
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

    # Plot distributions and boxplots
    plot_distributions(df, FIGURES_FOLDER)
    plot_boxplots(df, FIGURES_FOLDER)

    # Compute and plot correlation heatmap
    correlation_heatmap(
        df, FIGURES_FOLDER, os.path.join(OUTPUTS_FOLDER, "correlation_matrix.csv")
    )

    # Plot rolling volatility
    plot_rolling_volatility(df, FIGURES_FOLDER, window=30)

if __name__ == "__main__":
    main()