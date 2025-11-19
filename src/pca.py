"""
Compute principal components on daily returns, saves explained variance,
component loadings, and generates plots 
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

DATA_FILE = "results/data/returns.csv"
OUTPUTS_FOLDER = "results/outputs"
FIGURES_FOLDER = "results/figures"
N_COMPONENTS = 5  # keep 5 components


def load_returns(file_path: str) -> pd.DataFrame:
    """Load daily returns CSV as a pandas DataFrame."""
    return pd.read_csv(file_path, index_col=0, parse_dates=True)


def compute_pca(df: pd.DataFrame, n_components: int = N_COMPONENTS) -> PCA:
    """Fit PCA on the return matrix and return the PCA object."""
    pca = PCA(n_components=n_components)
    pca.fit(df.values)
    return pca


def save_pca_outputs(pca: PCA, df: pd.DataFrame, prefix: str = "pca") -> None:
    """Save explained variance and component loadings as CSVs."""
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

    # Explained variance ratio
    evr_df = pd.DataFrame(
        pca.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        columns=["explained_variance_ratio"]
    )
    evr_df.to_csv(os.path.join(OUTPUTS_FOLDER, f"{prefix}_explained_variance.csv"))

    # Component loadings
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    loadings_df.to_csv(os.path.join(OUTPUTS_FOLDER, f"{prefix}_loadings.csv"))


def plot_scree(pca: PCA, prefix: str = "pca") -> None:
    """Plot the explained variance ratio (scree plot)."""
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
             pca.explained_variance_ratio_, marker='o')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, f"{prefix}_scree.png"))
    plt.close()


def plot_loadings_heatmap(pca: PCA, df: pd.DataFrame, prefix: str = "pca") -> None:
    """Plot a heatmap of the PCA component loadings."""
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    plt.figure(figsize=(10,6))
    sns.heatmap(loadings_df, annot=True, cmap="coolwarm", center=0)
    plt.title("PCA Component Loadings")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, f"{prefix}_loadings_heatmap.png"))
    plt.close()


def main():
    df = load_returns(DATA_FILE)
    pca = compute_pca(df)
    save_pca_outputs(pca, df)
    plot_scree(pca)
    plot_loadings_heatmap(pca, df)


if __name__ == "__main__":
    main()