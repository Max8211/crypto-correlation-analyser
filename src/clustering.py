"""
Perform KMeans and Hierarchical clustering on correlation matrices.
Generates:
- Hierarchical dendrogram
- KMeans cluster labels + 2D scatter
- Silhouette score plot
- PCA scree plot
- Cluster-ordered correlation heatmap
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

OUTPUT_DIR = "results/outputs"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

CORR_PATH = os.path.join(OUTPUT_DIR, "rolling_corr_90d.csv")
OUTPUT_DIR2 = "results/data"
RETURNS_PATH = os.path.join(OUTPUT_DIR2, "returns.csv")


def load_correlation_matrix(path: str) -> pd.DataFrame:
    """Load rolling correlation CSV and produce symmetric coins x coins matrix"""
    df = pd.read_csv(path, index_col=[0,1])
    mean_corr = df.groupby(level=1).mean()
    mean_corr = (mean_corr + mean_corr.T) / 2
    return mean_corr


def corr_to_dist(corr: pd.DataFrame) -> np.ndarray:
    """Convert correlation matrix to distance matrix"""
    return np.sqrt(2 * (1 - corr))


def save_plot(fig, filename: str):
    """Helper to save figure and close it"""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {path}")


def run_hierarchical(dist: np.ndarray, labels):
    """Hierarchical clustering dendrogram"""
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=labels, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    fig.tight_layout()
    save_plot(fig, "hierarchical_dendrogram.png")
    return Z


def run_kmeans(corr: pd.DataFrame, k: int = 4) -> pd.Series:
    """Run KMeans and return cluster labels"""
    model = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = model.fit_predict(corr)
    return pd.Series(labels, index=corr.index)


def plot_kmeans_scatter(corr: pd.DataFrame, labels: pd.Series):
    """Plot 2D PCA scatter"""
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(corr)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in np.unique(labels):
        idx = labels == cluster
        ax.scatter(pcs[idx, 0], pcs[idx, 1], label=f"Cluster {cluster}", s=100)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("KMeans Clusters (PCA 2D)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    save_plot(fig, "kmeans_pca_scatter.png")


def plot_silhouette(corr: pd.DataFrame, max_k: int = 6):
    """Silhouette score plot for 2..max_k clusters"""
    scores = []
    ks = range(2, max_k + 1)
    for k in ks:
        model = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = model.fit_predict(corr)
        score = silhouette_score(corr, labels)
        scores.append(score)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, marker='o')
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs K")
    ax.grid(True)
    fig.tight_layout()
    save_plot(fig, "silhouette_score.png")


def plot_scree(returns: pd.DataFrame):
    """PCA scree plot for returns"""
    pca = PCA()
    pca.fit(returns)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(pca.explained_variance_ratio_)+1),
            pca.explained_variance_ratio_, marker='o')
    ax.set_title("PCA Scree Plot")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.grid(True)
    fig.tight_layout()
    save_plot(fig, "pca_scree_plot.png")


def plot_clustered_corr(corr: pd.DataFrame, labels: pd.Series):
    """Correlation heatmap ordered by cluster"""
    ordered_idx = labels.sort_values().index
    df = corr.loc[ordered_idx, ordered_idx]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    ax.set_title("Correlation Heatmap Ordered by Cluster")
    fig.tight_layout()
    save_plot(fig, "clustered_correlation_heatmap.png")


def save_cluster_labels(labels: pd.Series, name: str):
    """Save cluster labels"""
    path = os.path.join(OUTPUT_DIR, name)
    labels.to_csv(path)
    print(f"Saved: {path}")


def main():
    corr = load_correlation_matrix(CORR_PATH)
    dist = corr_to_dist(corr)

    print("Running hierarchical clustering...")
    run_hierarchical(dist, corr.index)

    print("Running KMeans clustering...")
    kmeans_labels = run_kmeans(corr, k=4)
    save_cluster_labels(kmeans_labels, "kmeans_clusters.csv")

    print("Plotting KMeans PCA scatter...")
    plot_kmeans_scatter(corr, kmeans_labels)

    print("Plotting silhouette score...")
    plot_silhouette(corr)

    print("Plotting PCA scree plot...")
    returns = pd.read_csv(RETURNS_PATH, index_col=0)
    plot_scree(returns)

    print("Plotting cluster-ordered correlation heatmap...")
    plot_clustered_corr(corr, kmeans_labels)

    print("Clustering analysis complete.")


if __name__ == "__main__":
    main()
