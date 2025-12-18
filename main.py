"""
Main script providing a summary of key metrics and results
Ensures full reproducibility by running the data pipeline before reporting.
"""

import os
import sys
import pandas as pd
import numpy as np
import subprocess

# --- PATHING FIX ---
# Define the project root and add BOTH root and src to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Now imports from src will work correctly
from src.regime_analysis import load_regime_outputs
from src.clustering import load_clustering_outputs
from src.supervised_regime_classification import run_supervised_regime_classification

# Define paths relative to the project root
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "outputs")
DATA_DIR = os.path.join(BASE_DIR, "results", "data")
CORR_PATH = os.path.join(OUTPUT_DIR, "rolling_corr_90d.csv")
REGIMES_PATH = os.path.join(OUTPUT_DIR, "detected_regimes.csv")
RETURNS_PATH = os.path.join(DATA_DIR, "returns.csv")

def run_pipeline():
    """
    Executes all data processing and analysis scripts in the correct order.
    """
    pipeline_scripts = [
        "merge_data.py",
        "returns.py",
        "eda.py",
        "correlation_analysis.py",
        "regime_analysis.py",
        "pca.py",
        "clustering.py"
    ]
    
    print("--- [System] Initializing Full Data Pipeline ---")
    for script in pipeline_scripts:
        script_path = os.path.join(SRC_DIR, script)
        print(f"Executing: {script}...")
        
        # Pass the current sys.path to the sub-process
        env = os.environ.copy()
        env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
        
        result = subprocess.run(
            [sys.executable, script_path], 
            capture_output=True, 
            text=True, 
            env=env
        )
        
        if result.returncode != 0:
            print(f"\n[CRITICAL ERROR] in {script}:")
            print(result.stderr)
            sys.exit(1)
    print("--- [System] Pipeline Complete. Generating Report Summary ---\n")

# Helper functions
def compute_overall_market_corr(corr_matrix):
    if corr_matrix.empty:
        return 0.0
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    return corr_matrix.where(mask).stack().mean()

def main():
    run_pipeline()

    print("=" * 60)
    print("Crypto Market Correlation Analysis")
    print("=" * 60)

    # 1. BTC analysis
    coin_names = []
    if os.path.exists(CORR_PATH):
        df = pd.read_csv(CORR_PATH, index_col=[0, 1])
        corr = df.groupby(level=1).mean()
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
        
        corr_offdiag = corr.where(~np.eye(corr.shape[0], dtype=bool))
        stacked = corr_offdiag.stack()

        if not stacked.empty:
            overall_avg_corr = stacked.mean()
            print("\n2. Overall Crypto-Market Correlation Summary")
            print(f"   • Average correlation across all coin-pairs: {overall_avg_corr:.2f}")

            if overall_avg_corr > 0.70:
                level = "very highly correlated."
            elif overall_avg_corr > 0.50:
                level = "highly correlated, therefore diversification opportunities are limited."
            else:
                level = "weakly correlated."
            print(f"   Interpretation: The crypto market is {level}")

    # 3. Market Regimes
    try:
        corr_normal, corr_stress, vol_normal, vol_stress, cluster_labels = load_regime_outputs()
        avg_norm = compute_overall_market_corr(corr_normal)
        avg_stress = compute_overall_market_corr(corr_stress)
        diff = avg_stress - avg_norm
        print("\n3. Market Regime Analysis (Normal vs Stress Periods)")
        print(f"   • Average normal-period correlation: {avg_norm:.2f}")
        print(f"   • Average stress-period correlation: {avg_stress:.2f}")
        print(f"   • Correlation increase during stress: {diff:+.2f}")
        print("Interpretation : Correlation rises sharply during market distress.")
    except Exception as e:
        print(f"\n[Error] Could not load regime outputs: {e}")

    # 4. Clustering
    try:
        pca_var, sil_score, kmeans_labels = load_clustering_outputs()
        if coin_names and not kmeans_labels.empty:
            kmeans_labels = kmeans_labels.reindex(coin_names)
        print("\n4. Machine Learning Analysis (K-Means Clustering)")
        valid_labels = kmeans_labels.dropna()
        unique_clusters = sorted(valid_labels.unique())
        print(f"   • Number of clusters: {len(unique_clusters)}")
        for cluster_id in unique_clusters:
            coins_in_cluster = valid_labels[valid_labels == cluster_id].index.tolist()
            print(f"     – Cluster {cluster_id} ({len(coins_in_cluster)} coins): {', '.join([c.capitalize() for c in coins_in_cluster])}")
        print(f"   Silhouette score: {sil_score:.3f}")
        if len(pca_var) >= 2:
            print(f"   • PCA Explained Var: PC1 {pca_var[0]:.2%}, PC2 {pca_var[1]:.2%}")
    except Exception as e:
        print(f"\n[Error] Could not load clustering outputs: {e}")

    # 5. Regime Classification
    print("\n5. Supervised Regime Classification (Random Forest)")
    print("-" * 60)
    try:
        run_supervised_regime_classification()
    except Exception as e:
        print(f"   [Error] Supervised script failed: {e}")
    print("-" * 60)
    print("Interpretation: Correlation serves as a reliable coincident indicator.")

if __name__ == "__main__":
    main()