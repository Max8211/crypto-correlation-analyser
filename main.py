"""
Main script providing a summary of key metrics and results
"""

import os
import pandas as pd
import numpy as np

from src.regime_analysis import load_regime_outputs
from src.clustering import load_clustering_outputs
from src.supervised_regime_classification import run_supervised_regime_classification

OUTPUT_DIR = "results/outputs"
DATA_DIR = "results/data"
CORR_PATH = os.path.join(OUTPUT_DIR, "rolling_corr_90d.csv")
REGIMES_PATH = os.path.join(OUTPUT_DIR, "detected_regimes.csv")
RETURNS_PATH = os.path.join(DATA_DIR, "returns.csv")


# Helper functions
def compute_overall_market_corr(corr_matrix):
    if corr_matrix.empty:
        return 0.0
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    return corr_matrix.where(mask).stack().mean()


def compute_cluster_summary(cluster_series):
    if cluster_series.empty:
        return {}
    return cluster_series.value_counts().to_dict()


def main():
    print(" ")
    print("=" * 60)
    print("Crypto Market Correlation Analysis")
    print("=" * 60)

    # 1. btc analysis
    coin_names = []
    if os.path.exists(CORR_PATH):
        df = pd.read_csv(CORR_PATH, index_col=[0, 1])
        corr = df.groupby(level=1).mean()
        # Ensure symmetric and consistent ordering
        corr = (corr + corr.T) / 2
        corr = corr.sort_index().sort_index(axis=1)
        coin_names = corr.columns.tolist()

        btc_name = ["bitcoin", "BTC", "btc"]
        btc_col = next((name for name in btc_name if name in corr.index), None)

        print("\n1. Bitcoin Correlation Summary")
        if btc_col:
            btc_corr = corr.loc[btc_col].drop(btc_col)
            print(f"   • Most correlated with BTC : {btc_corr.idxmax()} ({btc_corr.max():.2f})")
            print(f"   • Least correlated with BTC: {btc_corr.idxmin()} ({btc_corr.min():.2f})")
            print(f"   • Average BTC correlation with market: {btc_corr.mean():.2f}")
        else:
            print(f"   [Warning] Could not find Bitcoin in columns.")

        # 2.Overall market analysis
        corr_offdiag = corr.where(~np.eye(corr.shape[0], dtype=bool))
        stacked = corr_offdiag.stack()

        if not stacked.empty:
            overall_avg_corr = stacked.mean()
            print("\n2. Overall Crypto-Market Correlation Summary")
            print(f"   • Average correlation across all coin-pairs: {overall_avg_corr:.2f}")
            if overall_avg_corr > 0.70:
                level = " very highly correlated"
            elif overall_avg_corr > 0.50:
                level = "highly correlated, therefore diversification opportunities are limited."
            else:
                level = "weakly correlated"
            print(f"Interpretation: The crypto market is {level}.")
    else:
        print(f"\n[Warning] {CORR_PATH} not found. Skipping sections 1 & 2.")

    #3.Market Regimes
    try:
        corr_normal, corr_stress, vol_normal, vol_stress, cluster_labels = load_regime_outputs()

        avg_norm = compute_overall_market_corr(corr_normal)
        avg_stress = compute_overall_market_corr(corr_stress)
        diff = avg_stress - avg_norm

        diff_mat = corr_stress - corr_normal
        if not diff_mat.isnull().all().all():
            max_idx = np.nanargmax(diff_mat.values)
            min_idx = np.nanargmin(diff_mat.values)
            max_pair = np.unravel_index(max_idx, diff_mat.shape)
            min_pair = np.unravel_index(min_idx, diff_mat.shape)
            max_jump = diff_mat.index[max_pair[0]], diff_mat.columns[max_pair[1]], diff_mat.values[max_pair]
            min_jump = diff_mat.index[min_pair[0]], diff_mat.columns[min_pair[1]], diff_mat.values[min_pair]
        else:
            max_jump = ("N/A", "N/A", 0.0)
            min_jump = ("N/A", "N/A", 0.0)

        print("\n3. Market Regime Analysis (Normal vs Stress Periods)")
        print(f"   • Average normal-period correlation: {avg_norm:.2f}")
        print(f"   • Average stress-period correlation: {avg_stress:.2f}")
        print(f"   • Correlation increase during stress: {diff:+.2f}")
        print(f"   • Largest pairwise increase: {max_jump[0]} – {max_jump[1]} ({max_jump[2]:+.2f})")
        print("Interpretation : Correlation rises sharply during market distress.")
    except Exception as e:
        print(f"\n[Error] Could not load regime outputs: {e}")

    # 4. Clustering
    try:
        pca_var, sil_score, kmeans_labels = load_clustering_outputs()

        # If we have a rolling-corr coin order, align the labels to it.
        if coin_names and not kmeans_labels.empty:
            # Safe reindex: matches by coin name; leaves NaN where missing
            kmeans_labels = kmeans_labels.reindex(coin_names)
        # Now kmeans_labels.index matches coin_names order (if available)

        print("\n4. Machine Learning Analysis (K-Means Clustering)")
        # Remove NaNs when counting/printing cluster members
        valid_labels = kmeans_labels.dropna()
        unique_clusters = sorted(valid_labels.unique())
        print(f"   • Number of clusters: {len(unique_clusters)}")

        for cluster_id in unique_clusters:
            coins_in_cluster = valid_labels[valid_labels == cluster_id].index.tolist()
            coin_str = ", ".join([c.capitalize() for c in coins_in_cluster])
            print(f"     – Cluster {cluster_id} ({len(coins_in_cluster)} coins): {coin_str}")

        print("Interpretation: These clusters group assets that behave similarly. Picking coins from different clusters may help diversify risk.")

        if len(pca_var) >= 2:
            print("\n   PCA explained variance (first 2 PCs):")
            print(f"   • PC1: {pca_var[0]:.2%}, PC2: {pca_var[1]:.2%}, cumulative: {sum(pca_var[:2]):.2%}")
        print(f"   Silhouette score: {sil_score:.3f}")
        print("Interpretation: The low silhouette score indicates that clusters are not well-separated.")
    except Exception as e:
        print(f"\n[Error] Could not load clustering outputs: {e}")

    # 5. Regime Classification
    print("\n5. Supervised Regime Classification (Random Forest)")
    print("Can we classify the current market state using correlation metrics?")
    print("-" * 60)
    try:
        run_supervised_regime_classification()
    except Exception as e:
        print(f"   [Error] Supervised script failed: {e}")

    print("-" * 60)
    print("Interpretation: Correlation serves as a reliable coincident indicator.")
    print("The model shows high precision but is conservative (low recall) so it misses a great portion of stress periods.")
    print("This implies that high correlation is a sufficient metric for stress, however not a necessary one.")
    print(" ")


if __name__ == "__main__":
    main()