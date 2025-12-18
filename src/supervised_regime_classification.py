"""
Supervised Classification Script
Use fundamental correlation metrics to classify market regimes
"""

import os
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import correlation_analysis as correlation_script

warnings.filterwarnings("ignore", message="Could not infer format")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

OUTPUTS_DIR = "results/outputs"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

ROLLING_FILE = os.path.join(OUTPUTS_DIR, "rolling_corr_90d.csv")
EWMA_FILE = os.path.join(OUTPUTS_DIR, "ewma_corr_60d.csv")
REGIMES_FILE = os.path.join(OUTPUTS_DIR, "detected_regimes.csv")


def load_corr(file_path: str) -> pd.DataFrame:
    """Load correlation CSV and handle MultiIndex/Flat formats."""
    try:
        # Try loading as MultiIndex
        df = pd.read_csv(file_path, index_col=[0, 1], parse_dates=True)
        df = df.groupby(level=0).mean()
    except ValueError:
        # Fallback for flat format
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.ffill().bfill() 
    return df


def build_features(rolling: pd.DataFrame, ewma: pd.DataFrame) -> pd.DataFrame:
    """
    Compute logical features from correlation data
    """
    #Align indices
    common_idx = rolling.index.intersection(ewma.index)
    rolling = rolling.loc[common_idx]
    ewma = ewma.loc[common_idx]

    df = pd.DataFrame(index=common_idx)
    
    # Market Unity (Mean Correlation)
    df['rolling_mean'] = rolling.mean(axis=1)
    df['ewma_mean'] = ewma.mean(axis=1)
    
    # Momentum (Is correlation rising fast?)
    # If Rolling > EWMA, recent correlation is higher than the trend
    df['mean_diff'] = df['rolling_mean'] - df['ewma_mean']
    
    # Market Dispersion (Standard Deviation across coin pairs)
    df['corr_dispersion'] = rolling.std(axis=1)
    
    # Lags (Memory)
    # Give the model context of the last 3 days
    for lag in [1, 2, 3]:
        df[f'rolling_mean_lag{lag}'] = df['rolling_mean'].shift(lag)

    # velocity (1-Day Change) to determin if circulation is spiking
    
    df['corr_velocity'] = df['rolling_mean'] - df['rolling_mean_lag1']

    # Drop NaNs created by shift
    df = df.dropna()
    
    return df


def build_labels(features: pd.DataFrame) -> pd.Series:
    """Assign labels based on detected regimes."""
    if not os.path.exists(REGIMES_FILE):
        raise FileNotFoundError(f"Regimes file not found at {REGIMES_FILE}")

    regimes = pd.read_csv(REGIMES_FILE, parse_dates=['start', 'end'])
    labels = pd.Series(index=features.index, dtype="object")
    labels[:] = "normal"

    for _, row in regimes.iterrows():
        mask = (features.index >= row['start']) & (features.index <= row['end'])
        labels.loc[mask] = row['kind']
        
    return labels


def plot_feature_importance(clf: RandomForestClassifier, features: pd.DataFrame):
    if hasattr(clf, 'feature_importances_'):
        # Create the series
        imp = pd.Series(clf.feature_importances_, index=features.columns).sort_values(ascending=True)
        
        plt.figure(figsize=(10, 6))
        
        # Plot
        sns.barplot(x=imp.values, y=imp.index, hue=imp.index, palette="viridis", legend=False)
        
        plt.title("Feature Importance")
        
        plt.xlabel("Relative Importance") 
        plt.ylabel("Features")                 

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"))
        plt.close()

def load_supervised_outputs():
    """
    Check if required correlation files exist. If not, run correlation_analysis.py.
    """
    if not (os.path.exists(ROLLING_FILE) and os.path.exists(EWMA_FILE)):
        print("\n[System] Correlation files missing. Running src/correlation_analysis.py now...")
        correlation_script.main()
        print("[System] Correlation analysis complete. Resuming classification...\n")

def main():
    # Ensure dependency files are generated first
    load_supervised_outputs()

    # Continue with loading
    rolling = load_corr(ROLLING_FILE)
    ewma = load_corr(EWMA_FILE)

    features = build_features(rolling, ewma)
    labels = build_labels(features)

    # Split train/test (Time Series Split)
    # shuffle=False prevents data leakage (Train on Past, Test on Future)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, shuffle=False
    )

    # Class Weight Implementation
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    # set threshold 
    custom_threshold = 0.35

    # Get probability of "stress" class
    stress_col_index = list(clf.classes_).index("stress")
    y_probs = clf.predict_proba(X_test)[:, stress_col_index]

    # Apply custom threshold
    y_class_adjusted = np.where(y_probs >= custom_threshold, "stress", "normal")

    print(f"\nClassification Report")
    print(classification_report(y_test, y_class_adjusted))

    plot_feature_importance(clf, features)


def run_supervised_regime_classification():
    """Wrapper so main.py can call this script."""
    main()


if __name__ == "__main__":
    main()