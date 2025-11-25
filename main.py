"""
Main script for Crypto Market Correlation Analysis
Provides summary metrics and generates key figures.
"""

import os
import pandas as pd
import numpy as np

from src.cluster_regime_analysis import plot_clustered_corr, plot_regimes
from src.regime_analysis import main as regime_analysis_main

OUTPUT_DIR = "results/outputs"
FIG_DIR = "results/figures"
CORR_PATH = os.path.join(OUTPUT_DIR, "rolling_corr_90d.csv")
KMEANS_LABELS_PATH = os.path.join(OUTPUT_DIR, "kmeans_clusters.csv")

def main():
    print("="*60)
    print("Crypto Market Correlation Analysis")
    print("="*60)

    # Load correlation matrix (average over time)
    df = pd.read_csv(CORR_PATH, index_col=[0,1])
    corr = df.groupby(level=1).mean()
    corr = (corr + corr.T)/2

    # Bitcoin correlation summary
    btc_corr = corr.loc["bitcoin"].drop("bitcoin")
    most_corr_coin = btc_corr.idxmax()
    least_corr_coin = btc_corr.idxmin()
    avg_corr_btc = btc_corr.mean()

    print("\n1. Bitcoin Correlation Summary")
    print(f"   - Most correlated with BTC: {most_corr_coin} ({btc_corr[most_corr_coin]:.2f})")
    print(f"   - Least correlated with BTC: {least_corr_coin} ({btc_corr[least_corr_coin]:.2f})")
    print(f"   - Average correlation with BTC across all coins: {avg_corr_btc:.2f}")
    print("   - For full BTC correlations, refer to the clustered heatmap.")

    # Overall coin summary
    corr_values = corr.where(~np.eye(corr.shape[0], dtype=bool))  # mask diagonal
    max_pair_idx = np.unravel_index(np.nanargmax(corr_values.values), corr_values.shape)
    min_pair_idx = np.unravel_index(np.nanargmin(corr_values.values), corr_values.shape)
    max_pair = (corr_values.index[max_pair_idx[0]], corr_values.columns[max_pair_idx[1]])
    min_pair = (corr_values.index[min_pair_idx[0]], corr_values.columns[min_pair_idx[1]])
    max_corr = corr_values.values[max_pair_idx]
    min_corr = corr_values.values[min_pair_idx]

    print("\n2. Overall Coin Correlation Summary")
    print(f"   - Highest correlation overall: {max_pair[0]}-{max_pair[1]} ({max_corr:.2f})")
    print(f"   - Lowest correlation overall: {min_pair[0]}-{min_pair[1]} ({min_corr:.2f})")
    print("   - To see all 10 coins correlation, refer to the clustered heatmap.")

    # Generate clustered correlation heatmap
    if os.path.exists(KMEANS_LABELS_PATH):
        labels = pd.read_csv(KMEANS_LABELS_PATH, index_col=0).iloc[:,0]
        plot_clustered_corr(corr, labels)
        print("   - Clustered heatmap generated in results/figures/")
 
    # Run regime analysis and generate rolling correlation plots
    print("\n3. Rolling Correlation Regimes")
    plot_regimes(CORR_PATH, ("bitcoin","ethereum"))
    plot_regimes(CORR_PATH, ("bitcoin","solana"))
    print("   - Rolling correlation plots saved in results/figures/")
    print("   - For full stress vs normal period analysis, see regime analysis figures.")

    # Optionally run full regime analysis (creates stress/normal windows, vol, cluster stability)
    print("\nRunning regime analysis...")
    regime_analysis_main()
    print("All analysis complete. Check results/figures for plots and results/outputs for data tables.")

if __name__ == "__main__":
    main()