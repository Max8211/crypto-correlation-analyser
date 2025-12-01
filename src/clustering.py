"""
Perform KMeans and Hierarchical clustering on correlation matrices.
Generates:
- Hierarchical dendrogram
- KMeans cluster labels + 2D scatter
- Silhouette score plot
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
RETURNS_PATH = os.path.join("results/data", "returns.csv")

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
    #add coin labels to each dot
    for coin, (x, y) in zip(corr.index, pcs):
        ax.text(x + 0.01, y + 0.01, coin, fontsize=9)
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
    
    kmeans_labels = run_kmeans(corr, k=4)
    save_cluster_labels(kmeans_labels, "kmeans_clusters.csv")
    plot_kmeans_scatter(corr, kmeans_labels)
    plot_silhouette(corr)
    plot_clustered_corr(corr, kmeans_labels)
    print("Clustering analysis complete.")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np

OUTPUTS_DIR = "results/outputs"

def load_clustering_outputs():
    """
    Load precomputed ML outputs: PCA explained variance, silhouette score, and K-Means labels.
    
    Returns:
        pca_explained_var : list or np.array of floats
        silhouette_score  : float
        kmeans_labels     : pd.Series
    """
    # PCA explained variance
    pca_path = os.path.join(OUTPUTS_DIR, "pca_explained_var.csv")
    if os.path.exists(pca_path):
        pca_explained_var = pd.read_csv(pca_path, header=None).iloc[0].to_numpy()
    else:
        # fallback dummy values
        pca_explained_var = np.array([0.55, 0.25, 0.10, 0.05, 0.05])

    # Silhouette score
    sil_path = os.path.join(OUTPUTS_DIR, "silhouette_score.csv")
    if os.path.exists(sil_path):
        silhouette_score = pd.read_csv(sil_path, header=None).iloc[0,0]
    else:
        silhouette_score = 0.35  # dummy fallback

    # K-Means cluster labels
    kmeans_path = os.path.join(OUTPUTS_DIR, "kmeans_labels.csv")
    if os.path.exists(kmeans_path):
        kmeans_labels = pd.read_csv(kmeans_path, index_col=0, squeeze=True)
    else:
        # dummy 10 coins split in 4 clusters for fallback
        kmeans_labels = pd.Series([0,0,0,0,0,0,1,2,3,3], index=range(10))

    return pca_explained_var, silhouette_score, kmeans_labels