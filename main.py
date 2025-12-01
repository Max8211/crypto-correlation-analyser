"""
Main script for Crypto Market Correlation Analysis
Compute relevant results metrics for correlations, regimes, and clustering results.
"""

import os
import pandas as pd
import numpy as np

from src.regime_analysis import load_regime_outputs   # must return: corr_normal, corr_stress, vol_normal, vol_stress, cluster_labels
from src.clustering import load_clustering_outputs    # must return: pca_explained_var, silhouette_score, kmeans_labels

OUTPUT_DIR = "results/outputs"
CORR_PATH = os.path.join(OUTPUT_DIR, "rolling_corr_90d.csv")

#Helper functions
def compute_overall_market_corr(corr_matrix):
    """Return average correlation across all off-diagonal entries."""
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    return corr_matrix.where(mask).stack().mean()


def compute_cluster_summary(cluster_series):
    """Return cluster counts as dictionary."""
    return cluster_series.value_counts().to_dict()

# Main analysis script

def main():
    print("="*60)
    print("Crypto Market Correlation Analysis")
    print("="*60)

    # 1. Load correlation matrix
    df = pd.read_csv(CORR_PATH, index_col=[0, 1])
    corr = df.groupby(level=1).mean()
    corr = (corr + corr.T) / 2  # enforce symmetry

    # 1. Bitcoin correlation summary--
    btc_corr = corr.loc["bitcoin"].drop("bitcoin")
    print("\n1. Bitcoin Correlation Summary")
    print(f"   • Most correlated with BTC : {btc_corr.idxmax()} ({btc_corr.max():.2f})")
    print(f"   • Least correlated with BTC: {btc_corr.idxmin()} ({btc_corr.min():.2f})")
    print(f"   • Average BTC correlation with market: {btc_corr.mean():.2f}")

    # 2. Overall market correlation summary
    corr_offdiag = corr.where(~np.eye(corr.shape[0], dtype=bool))
    stacked = corr_offdiag.stack()  # MultiIndex series of all off-diagonal correlations

    highest_pair_idx = stacked.idxmax()
    highest_val = stacked.max()
    lowest_pair_idx = stacked.idxmin()
    lowest_val = stacked.min()
    overall_avg_corr = stacked.mean()

    print("\n2. Overall Crypto-Market Correlation Summary")
    print(f"   • Highest correlation overall: {highest_pair_idx[0]} – {highest_pair_idx[1]} ({highest_val:.2f})")
    print(f"   • Lowest correlation overall: {lowest_pair_idx[0]} – {lowest_pair_idx[1]} ({lowest_val:.2f})")
    print(f"   • Average correlation across all coin-pairs: {overall_avg_corr:.2f}")

    # qualitative interpretation
    if overall_avg_corr > 0.70:
        level = "highly correlated"
    elif overall_avg_corr > 0.50:
        level = "moderately correlated"
    else:
        level = "weakly correlated"

    print(f"   → Interpretation: The major 10-coin crypto market is {level}, diversification benefits exist but are modest and limited. ")

    # 3.Market Regimes 
    corr_normal, corr_stress, vol_normal, vol_stress, cluster_labels = load_regime_outputs()

    avg_norm = compute_overall_market_corr(corr_normal)
    avg_stress = compute_overall_market_corr(corr_stress)
    diff = avg_stress - avg_norm

    # largest/smallest pairwise correlation change
    diff_mat = corr_stress - corr_normal
    max_idx = np.nanargmax(diff_mat.values)
    min_idx = np.nanargmin(diff_mat.values)
    max_pair = np.unravel_index(max_idx, diff_mat.shape)
    min_pair = np.unravel_index(min_idx, diff_mat.shape)
    max_jump = diff_mat.index[max_pair[0]], diff_mat.columns[max_pair[1]], diff_mat.values[max_pair]
    min_jump = diff_mat.index[min_pair[0]], diff_mat.columns[min_pair[1]], diff_mat.values[min_pair]

    print("\n3. Market Regime Analysis (Normal vs Stress Periods)")
    print(f"   • Average normal-period correlation: {avg_norm:.2f}")
    print(f"   • Average stress-period correlation: {avg_stress:.2f}")
    print(f"   • Correlation increase during stress: {diff:.2f}")
    print(f"   • Largest pairwise increase: {max_jump[0]} – {max_jump[1]} ({max_jump[2]:+.2f})")
    print(f"   • Smallest pairwise change: {min_jump[0]} – {min_jump[1]} ({min_jump[2]:+.2f})")
    print("     Correlations rise sharply during market distress.")
    
    # 4. Machine Learning
    pca_var, sil_score, kmeans_labels = load_clustering_outputs()
    cluster_summary = compute_cluster_summary(kmeans_labels)

    print("\n4. Machine Learning Analysis (K-Means Clustering)")
    print(f"   • Number of clusters detected: {len(cluster_summary)}")
    for cl, count in cluster_summary.items():
        print(f"     – Cluster {cl}: {count} windows")

    print("   • Interpretation: Some clusters are small, reflecting coins with exceptional behavior compared to typical market dynamics.")

    #  add PCA and silhouette
    print("\n   PCA explained variance (first 2 PCs):")
    print(f"   • PC1: {pca_var[0]:.2%}, PC2: {pca_var[1]:.2%}, cumulative: {sum(pca_var[:2]):.2%}")
    print(f"   Silhouette score: {sil_score:.3f}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()