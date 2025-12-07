"""
Compute and visualize rolling and EWMA correlations 
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
    """
    Compute rolling correlation matrix.
    """
    return df.rolling(window).corr().dropna()


def ewma_correlation(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """
    Compute EWMA correlation matrix
    """
    return df.ewm(span=span).corr().dropna()


def extract_pairwise_data(multi_index_corr: pd.DataFrame, pairs: List[tuple]) -> pd.DataFrame:
    """
    Helper: Extracts specific pairs (e.g., BTC-ETH) from the MultiIndex matrix
    into a flat DataFrame for plotting.
    """
    dates = multi_index_corr.index.get_level_values(0).unique()
    pair_df = pd.DataFrame(index=dates)

    for c1, c2 in pairs:
        try:
            # Get all rows where level-1 index is c1, then select column c2
            series = multi_index_corr.xs(c1, level=1)[c2]
            pair_df[f"{c1}-{c2}"] = series
        except KeyError:
            print(f"Warning: Pair {c1}-{c2} not found in data.")
            continue
            
    return pair_df


def plot_selected_pairs(pair_df: pd.DataFrame, figures_folder: str, title: str):
    """Plot correlation for selected key coin pairs"""
    plt.figure(figsize=(12, 6))
    for col in pair_df.columns:
        plt.plot(pair_df.index, pair_df[col].values, label=col.upper())
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + "_pairs.png"
    plt.savefig(os.path.join(figures_folder, fname))
    plt.close()


def main():
    """Load returns, compute rolling/EWMA correlations, and plot results"""
    print("Computing correlations...")
    returns = load_returns(DATA_FILE)

    # 1. Rolling correlations
    rolling_corr = rolling_correlation(returns, ROLLING_WINDOW)
    rolling_corr.to_csv(os.path.join(OUTPUTS_FOLDER, f"rolling_corr_{ROLLING_WINDOW}d.csv"))
    
    # Process for local plots (Pairs Only)
    rolling_pairs = extract_pairwise_data(rolling_corr, KEY_PAIRS)
    plot_selected_pairs(rolling_pairs, FIGURES_FOLDER, f"Rolling {ROLLING_WINDOW}-Day")
    
    # 2. EWMA correlations
    ewma_corr = ewma_correlation(returns, EWMA_SPAN)
    ewma_corr.to_csv(os.path.join(OUTPUTS_FOLDER, f"ewma_corr_{EWMA_SPAN}d.csv"))
    
    # Process for local plots (Pairs Only)
    ewma_pairs = extract_pairwise_data(ewma_corr, KEY_PAIRS)
    plot_selected_pairs(ewma_pairs, FIGURES_FOLDER, f"EWMA {EWMA_SPAN}-Day")
    
    # NOTE: "Mean" and "Regime Shaded" plots have been removed as requested.
    
    print("Correlation analysis complete.")


if __name__ == "__main__":
    main()