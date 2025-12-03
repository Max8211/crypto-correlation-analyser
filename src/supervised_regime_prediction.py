"""
Use rolling and EWMA correlation datasets, computes features,
and trains a RandomForestClassifier to classify 'stress' vs 'normal' periods.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUTS_DIR = "results/outputs"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

ROLLING_FILE = os.path.join(OUTPUTS_DIR, "rolling_corr_90d.csv")
EWMA_FILE = os.path.join(OUTPUTS_DIR, "ewma_corr_60d.csv")
REGIMES_FILE = os.path.join(OUTPUTS_DIR, "detected_regimes.csv")


def load_corr(file_path: str) -> pd.DataFrame:
    """Load correlation CSV and ensure numeric values only"""
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(method='ffill').fillna(method='bfill')  # fill missing values
    return df


def build_features(rolling: pd.DataFrame, ewma: pd.DataFrame) -> pd.DataFrame:
    """Compute features from rolling and EWMA correlations"""
    df = pd.DataFrame(index=rolling.index)
    # mean correlation across all coins
    df['rolling_mean'] = rolling.mean(axis=1)
    df['ewma_mean'] = ewma.mean(axis=1)
    # mean difference
    df['mean_diff'] = df['rolling_mean'] - df['ewma_mean']
    # rolling std
    df['rolling_std'] = rolling.std(axis=1)
    df['ewma_std'] = ewma.std(axis=1)
    return df


def build_labels(features: pd.DataFrame) -> pd.Series:
    """Assign labels based on detected regimes."""
    regimes = pd.read_csv(REGIMES_FILE, parse_dates=['start', 'end'])
    labels = pd.Series(index=features.index, dtype="object")
    for _, row in regimes.iterrows():
        mask = (features.index >= row['start']) & (features.index <= row['end'])
        labels.loc[mask] = row['kind']
    return labels.fillna("normal")  # any missing default to 'normal'


def plot_feature_importance(clf: RandomForestClassifier, features: pd.DataFrame):
    imp = pd.Series(clf.feature_importances_, index=features.columns).sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, palette="viridis")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"))
    plt.close()
    print(f"Feature importance plot saved to {os.path.join(FIGURES_DIR, 'feature_importance.png')}")


def main():
    # Load data
    rolling = load_corr(ROLLING_FILE)
    ewma = load_corr(EWMA_FILE)

    # Build features & labels
    features = build_features(rolling, ewma)
    labels = build_labels(features)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict & report
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Feature importance
    plot_feature_importance(clf, features)

    print("Supervised regime prediction complete. Outputs saved.")


if __name__ == "__main__":
    main()
