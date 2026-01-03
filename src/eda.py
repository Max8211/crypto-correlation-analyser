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


def correlation_heatmap(
    df: pd.DataFrame, figures_folder: str, output_path: str
) -> None:
    """Compute correlation matrix"""
    corr = df.corr()
    corr.to_csv(output_path)


def plot_rolling_volatility(
    df: pd.DataFrame, figures_folder: str, window: int = 30
) -> None:
    """Plot rolling volatility with clipping and log scale for readability."""
    rolling_vol = df.rolling(window=window).std()


def main():
    df = load_returns(DATA_FILE)

    # Compute and save descriptive statistics (mean, skew, kurtosis, etc.)
    summary_statistics(df, os.path.join(OUTPUTS_FOLDER, "summary_statistics.csv"))

    # Compute and save the correlation matrix CSV
    correlation_heatmap(
        df, FIGURES_FOLDER, os.path.join(OUTPUTS_FOLDER, "correlation_matrix.csv")
    )


if __name__ == "__main__":
    main()
