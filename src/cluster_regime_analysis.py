"""
Generate cluster-ordered correlation heatmap and rolling 90-day correlation regimes.
This version avoids rerunning KMeans, PCA, silhouette, etc., assuming they are already done.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "results/outputs"
FIG_DIR = "results/figures"
CORR_PATH = os.path.join(OUTPUT_DIR, "rolling_corr_90d.csv")
KMEANS_LABELS_PATH = os.path.join(OUTPUT_DIR, "kmeans_clusters.csv")

os.makedirs(FIG_DIR, exist_ok=True)

def load_correlation_matrix(path: str) -> pd.DataFrame:
    """Load rolling correlations and average over time"""
    df = pd.read_csv(path, index_col=[0,1])
    mean_corr = df.groupby(level=1).mean()
    mean_corr = (mean_corr + mean_corr.T)/2
    return mean_corr

def plot_clustered_corr(corr: pd.DataFrame, labels: pd.Series):
    """Correlation heatmap ordered by cluster"""
    ordered_idx = labels.sort_values().index
    df = corr.loc[ordered_idx, ordered_idx]
    plt.figure(figsize=(10,8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap Ordered by Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "clustered_corr_heatmap.png"))
    plt.close()

def plot_regimes(rolling_corr_path: str, coin_pair: tuple=("bitcoin","ethereum"), window:int=90):
    """Plot rolling correlation between two coins over time with high/low correlation highlighting"""
    df = pd.read_csv(rolling_corr_path, index_col=0, parse_dates=True)

    # Check coins exist
    if coin_pair[0] not in df.columns or coin_pair[1] not in df.columns:
        raise ValueError(f"One of the coins {coin_pair} not in CSV columns.")
    
    # Compute rolling correlation
    series = df[coin_pair[0]].rolling(window).corr(df[coin_pair[1]])

    plt.figure(figsize=(12,6))
    plt.plot(series.index, series.values, label=f"{coin_pair[0].capitalize()}-{coin_pair[1].capitalize()} rolling correlation")
    plt.axhline(series.mean(), color='red', linestyle='--', label='Mean Correlation')

    # Exact same highlighting as before
    plt.fill_between(series.index, -1, 1, where=(series>0.8), color='green', alpha=0.1, label="High correlation")
    plt.fill_between(series.index, -1, 1, where=(series<0), color='yellow', alpha=0.3, label="Low correlation")

    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.title(f"{coin_pair[0].capitalize()}-{coin_pair[1].capitalize()} {window}-Day Rolling Correlation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,f"{coin_pair[0]}_{coin_pair[1]}_rolling_corr.png"))
    plt.close()

def main():
    # Load KMeans labels (already computed in clustering script)
    if not os.path.exists(KMEANS_LABELS_PATH):
        raise FileNotFoundError("KMeans labels CSV not found. Run clustering script first.")
    
    kmeans_labels = pd.read_csv(KMEANS_LABELS_PATH, index_col=0)
    kmeans_labels = kmeans_labels.iloc[:, 0]  # convert single-column DataFrame to Series

    # Load correlation matrix
    corr = load_correlation_matrix(CORR_PATH)

    print("Plotting cluster-ordered correlation heatmap...")
    plot_clustered_corr(corr, kmeans_labels)

    print("Plotting rolling correlation regimes...")
    plot_regimes(CORR_PATH, ("bitcoin","ethereum"))
    plot_regimes(CORR_PATH, ("bitcoin","solana"))

    print("Cluster + Regime analysis complete.")

if __name__=="__main__":
    main()
