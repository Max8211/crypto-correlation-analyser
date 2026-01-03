"""
Clustering Script
Perform KMeans clustering, create plots,
and save PCA, silhouette, and clustering outputs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

OUTPUTS_DIR = "results/outputs"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# PATHS
RETURNS_PATH = os.path.join("results/data", "returns.csv")


def load_returns(path: str) -> pd.DataFrame:
    """Load the returns.csv dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Ensure returns.csv is in results/data."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def save_plot(fig, filename: str):
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def run_kmeans(feature_df: pd.DataFrame, k: int = 2) -> pd.Series:
    """
    Run KMeans on the feature matrix (rows=coins, columns=features)
    """
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(feature_df.values)
    return pd.Series(labels, index=feature_df.index, name="cluster")


def plot_mds_scatter(corr: pd.DataFrame, labels: pd.Series, explained_var: np.ndarray):
    """
    Plot 2D Scatter using MDS (Multidimensional Scaling).
    """
    # 1.Convert Correlation to Distance (Dissimilarity)
    dist_matrix = 1 - corr.values

    # 2. Run MDS on the precomputed dissimilarity
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(dist_matrix)
    coords_df = pd.DataFrame(coords, index=corr.index, columns=["Dim1", "Dim2"])

    # 3. Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = np.unique(labels)

    # colors
    tab10 = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))

    for i, cluster in enumerate(unique_labels):
        idx = labels == cluster
        cluster_coords = coords_df.loc[idx]
        ax.scatter(
            cluster_coords["Dim1"],
            cluster_coords["Dim2"],
            label=f"Cluster {cluster}",
            s=150,
            color=tab10[i],
            alpha=0.85,
            edgecolors="k",
        )

        # Annotate points
    offset_x = 0.02 * (coords_df["Dim1"].max() - coords_df["Dim1"].min())
    offset_y = 0.02 * (coords_df["Dim2"].max() - coords_df["Dim2"].min())

    for coin in coords_df.index:
        x, y = coords_df.loc[coin]
        ax.text(
            x + offset_x,
            y + offset_y,
            str(coin).capitalize(),
            fontsize=9,
            weight="bold",
        )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    title_var = (
        explained_var.sum()
        if isinstance(explained_var, (list, np.ndarray))
        else explained_var
    )
    ax.set_title(f"Correlation Map")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    save_plot(fig, "kmeans_pca_scatter.png")


def plot_silhouette(feature_df: pd.DataFrame, max_k: int = 6):
    """
    Plot silhouette scores for k=2 to max_k to help choose optimal cluster count.
    """
    scores = []
    ks = range(2, max_k + 1)
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(feature_df.values)
        score = silhouette_score(feature_df.values, labels)
        scores.append(score)
    # Plot silhouette curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Optimal Cluster Count (Silhouette)")
    ax.grid(True)
    fig.tight_layout()
    save_plot(fig, "silhouette_score.png")


def main():
    print("Loading Data...")
    try:
        # Load data and calculate the static correlation matrix
        returns = load_returns(RETURNS_PATH)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return
    # Sort indices to ensure consistent coin mapping across all scripts
    static_corr = returns.corr()
    static_corr = static_corr.sort_index().sort_index(axis=1)

    order_path = os.path.join(OUTPUTS_DIR, "clustering_coin_order.csv")
    pd.Series(static_corr.columns).to_csv(order_path, index=False)
    print(f"Saved coin order to {order_path}")

    # PCA: Reduce dimensions to capture major variance components
    print("Computing PCA Stats...")
    n_comp = min(2, min(returns.shape))
    pca_stats = PCA(n_components=n_comp)
    pca_stats.fit(returns.fillna(0))
    pca_var_path = os.path.join(OUTPUTS_DIR, "pca_explained_var.csv")
    np.savetxt(pca_var_path, pca_stats.explained_variance_ratio_, delimiter=",")
    print(f"Saved PCA Variance to {pca_var_path}")

    # KMeans Clustering
    print("Running KMeans...")
    kmeans_labels = run_kmeans(static_corr, k=2)
    kmeans_path = os.path.join(OUTPUTS_DIR, "kmeans_clusters.csv")
    kmeans_labels.to_csv(kmeans_path, header=True)
    print(f"Saved KMeans clusters to {kmeans_path}")
    # Compute silhouette score for chosen K
    sil_score = silhouette_score(static_corr.values, kmeans_labels.values)
    sil_path = os.path.join(OUTPUTS_DIR, "silhouette_score.csv")
    with open(sil_path, "w") as f:
        f.write(str(float(sil_score)))
    print(f"Saved Silhouette Score: {sil_score:.3f} -> {sil_path}")

    plot_mds_scatter(static_corr, kmeans_labels, pca_stats.explained_variance_ratio_)
    plot_silhouette(static_corr)

    print("Clustering analysis complete.")


def load_clustering_outputs():
    """
    Load PCA variance, silhouette score, and KMeans labels from saved outputs."""
    pca_path = os.path.join(OUTPUTS_DIR, "pca_explained_var.csv")
    sil_path = os.path.join(OUTPUTS_DIR, "silhouette_score.csv")
    kmeans_path = os.path.join(OUTPUTS_DIR, "kmeans_clusters.csv")
    # Dependency check: Automatically run analysis if files are missing
    if not (
        os.path.exists(pca_path)
        and os.path.exists(sil_path)
        and os.path.exists(kmeans_path)
    ):
        print(
            "\n[System] Clustering output files missing. Running src/clustering.py now..."
        )
        main()
        print("[System] Clustering complete. Resuming main script...\n")

    # Load and format PCA explained variance
    if os.path.exists(pca_path):
        pca_explained_var = np.loadtxt(pca_path, delimiter=",")
        if pca_explained_var.ndim == 0:
            pca_explained_var = np.array([pca_explained_var])
    else:
        pca_explained_var = np.array([0.0, 0.0])

    # Load and format Silhouette score
    if os.path.exists(sil_path):
        with open(sil_path, "r") as f:
            val = f.read().strip()
            silhouette_score_val = float(val) if val else 0.0
    else:
        silhouette_score_val = 0.0

    # Load and format K-Means labels
    if os.path.exists(kmeans_path):
        kmeans_labels = pd.read_csv(kmeans_path, index_col=0).iloc[:, 0]
        kmeans_labels = pd.Series(kmeans_labels, name="cluster")
    else:
        kmeans_labels = pd.Series(dtype=float)

    return pca_explained_var, silhouette_score_val, kmeans_labels


if __name__ == "__main__":
    main()
