"""
Compute and save rolling and EWMA correlations.
Visual plotting code has been removed to keep the results folder clean.
"""

import os
import pandas as pd

DATA_FILE = "results/data/returns.csv"
OUTPUTS_FOLDER = "results/outputs"

# Ensure output directory exists
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

# Parameters used for feature construction
ROLLING_WINDOW = 90
EWMA_SPAN = 60

def load_returns(file_path: str) -> pd.DataFrame:
    """Load daily returns into a DataFrame with datetime index."""
    return pd.read_csv(file_path, index_col=0, parse_dates=True)


def rolling_correlation(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling correlation matrix.
    [cite_start]Used as input for the Random Forest classifier[cite: 109].
    """
    return df.rolling(window).corr().dropna()


def ewma_correlation(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """
    Compute EWMA correlation matrix.
    [cite_start]Used to identify absolute strength of correlation[cite: 364].
    """
    return df.ewm(span=span).corr().dropna()


def main():
    """Load returns and compute rolling/EWMA correlations."""
    print("Computing correlations...")
    
    # Check if data file exists to prevent pathing errors
    if not os.path.exists(DATA_FILE):
        print(f"[Error] {DATA_FILE} not found. Ensure returns.py has run.")
        return

    returns = load_returns(DATA_FILE)

    # 1. Rolling correlations: Required for 'rolling_mean' feature
    rolling_corr = rolling_correlation(returns, ROLLING_WINDOW)
    rolling_corr.to_csv(os.path.join(OUTPUTS_FOLDER, f"rolling_corr_{ROLLING_WINDOW}d.csv"))
    
    # 2. EWMA correlations: Required for 'ewma_mean' feature
    ewma_corr = ewma_correlation(returns, EWMA_SPAN)
    ewma_corr.to_csv(os.path.join(OUTPUTS_FOLDER, f"ewma_corr_{EWMA_SPAN}d.csv"))
    
    print("Correlation analysis complete. Data saved to results/outputs.")


if __name__ == "__main__":
    main()