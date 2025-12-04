"""
Compute and visualize rolling and EWMA correlations for major cryptocurrencies.
FIXED: Saves full MultiIndex correlation matrices so main.py can read them correctly.
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
    RETURNS: MultiIndex DataFrame (Date, Ticker) -> Full N x N matrix.
    """
    # We dropna() to remove the empty initial window
    return df.rolling(window).corr().dropna()


def ewma_correlation(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """
    Compute EWMA correlation matrix.
    RETURNS: MultiIndex DataFrame (Date, Ticker) -> Full N x N matrix.
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
        # We look up: For every Date, look at Coin1 (level 1), and get value of Coin2 (column)
        # Using xs (cross-section) is efficient here
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


def plot_average_correlation(corr_means: pd.DataFrame, figures_folder: str, title: str):
    """
    Plot mean correlation across all coins.
    corr_means should be (Date x Ticker) where values are avg correlation of that ticker.
    """
    plt.figure(figsize=(12, 6))
    # Mean of the means = Overall Market Correlation
    market_avg = corr_means.mean(axis=1)
    
    plt.plot(market_avg.index, market_avg.values, color="navy")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Mean Correlation")
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + "_mean.png"
    plt.savefig(os.path.join(figures_folder, fname))
    plt.close()


def detect_correlation_regimes(corr_means: pd.DataFrame, title: str, threshold_std: float = 1.0):
    """
    Identify high/low correlation regimes based on Market Average.
    """
    # Overall market correlation index
    mean_corr = corr_means.mean(axis=1)
    
    mean_val = mean_corr.mean()
    std_val = mean_corr.std()
    high_thresh = mean_val + threshold_std * std_val
    low_thresh = mean_val - threshold_std * std_val

    regimes = pd.Series(index=mean_corr.index, dtype="object")
    regimes[:] = "normal"
    regimes[mean_corr >= high_thresh] = "high"
    regimes[mean_corr <= low_thresh] = "low"

    # Save
    regimes_df = pd.DataFrame({"mean_correlation": mean_corr, "regime": regimes})
    csv_path = os.path.join(OUTPUTS_FOLDER, f"{title.lower().replace(' ','_')}_regimes.csv")
    regimes_df.to_csv(csv_path)

    # Plot shaded regions
    plt.figure(figsize=(12,6))
    plt.plot(mean_corr.index, mean_corr.values, color="navy", label="Mean Correlation")
    
    # Shade high regimes
    high_dates = regimes[regimes == "high"].index
    if len(high_dates) > 0:
        for i in high_dates:
             plt.axvspan(i, i, color="red", alpha=0.1, linewidth=0)
             
    # Shade low regimes
    low_dates = regimes[regimes == "low"].index
    if len(low_dates) > 0:
        for i in low_dates:
             plt.axvspan(i, i, color="green", alpha=0.1, linewidth=0)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Mean Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, f"{title.lower().replace(' ','_')}_shaded.png"))
    plt.close()


def main():
    """Load returns, compute rolling/EWMA correlations, and plot results"""
    print("Computing correlations...")
    returns = load_returns(DATA_FILE)

    # 1. Rolling correlations
    # Returns FULL MultiIndex Matrix (Date, Ticker) -> This fixes main.py!
    rolling_corr = rolling_correlation(returns, ROLLING_WINDOW)
    rolling_corr.to_csv(os.path.join(OUTPUTS_FOLDER, f"rolling_corr_{ROLLING_WINDOW}d.csv"))
    
    # Process for local plots
    # Flatten specific pairs
    rolling_pairs = extract_pairwise_data(rolling_corr, KEY_PAIRS)
    plot_selected_pairs(rolling_pairs, FIGURES_FOLDER, f"Rolling {ROLLING_WINDOW}-Day")
    
    # Collapse for mean analysis
    rolling_means = rolling_corr.groupby(level=0).mean()
    plot_average_correlation(rolling_means, FIGURES_FOLDER, f"Rolling {ROLLING_WINDOW}-Day")
    detect_correlation_regimes(rolling_means, f"Rolling {ROLLING_WINDOW}-Day Regimes")

    # 2. EWMA correlations
    ewma_corr = ewma_correlation(returns, EWMA_SPAN)
    ewma_corr.to_csv(os.path.join(OUTPUTS_FOLDER, f"ewma_corr_{EWMA_SPAN}d.csv"))
    
    # Process for local plots
    ewma_pairs = extract_pairwise_data(ewma_corr, KEY_PAIRS)
    plot_selected_pairs(ewma_pairs, FIGURES_FOLDER, f"EWMA {EWMA_SPAN}-Day")
    
    # Collapse for mean analysis
    ewma_means = ewma_corr.groupby(level=0).mean()
    plot_average_correlation(ewma_means, FIGURES_FOLDER, f"EWMA {EWMA_SPAN}-Day")
    detect_correlation_regimes(ewma_means, f"EWMA {EWMA_SPAN}-Day Regimes")
    
    print("Correlation analysis complete.")


if __name__ == "__main__":
    main()