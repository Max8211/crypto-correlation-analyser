"""
Detect stress windows using Downside Volatility.
Threshold: Top 15% days
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RETURNS_PATH = "results/data/returns.csv"
OUTPUTS_DIR = "results/outputs"
FIGURES_DIR = "results/figures"

# volatility parameters
VOL_WINDOW = 14 
# Top 15% of downside volatility days
VOL_PERCENTILE = 0.85

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_returns(path: str) -> pd.DataFrame:
   df = pd.read_csv(path, index_col=0, parse_dates=True)
   return df


def build_market_index(returns: pd.DataFrame) -> pd.Series:
   eq_ret = returns.mean(axis=1)
   idx = (1 + eq_ret).cumprod()
   idx.name = "market_index"
   return idx


def compute_downside_volatility(returns: pd.DataFrame, window: int) -> pd.Series:
   market_ret = returns.mean(axis=1)
   neg_ret = market_ret.copy()
   neg_ret[neg_ret > 0] = 0
  
   downside_vol = neg_ret.rolling(window=window).std()
   downside_vol.name = "downside_volatility"
   return downside_vol


def detect_stress_periods(stress_mask: pd.Series,
                          min_len: int = 2) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Identify continuous periods where stress_mask is True.
    """
    windows = []
    in_window = False
    start = None
    prev_ts = None
   
    # Ensure the series is boolean and drop NaNs (start of data)
    clean_series = stress_mask.dropna()

    for ts, is_stress in clean_series.items():
        if is_stress and not in_window:
            in_window = True
            start = ts
        elif not is_stress and in_window:
            end = prev_ts
            if (end - start).days + 1 >= min_len:
                windows.append((start, end))
            in_window = False
        prev_ts = ts
       
    if in_window:
        end = prev_ts
        if (end - start).days + 1 >= min_len:
            windows.append((start, end))
    return windows


def pick_representative_period(windows: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> Tuple[pd.Timestamp, pd.Timestamp]:
   if not windows:
       raise ValueError("No stress periods found")
   lengths = [(win[1] - win[0]).days + 1 for win in windows]
   idx = int(np.argmax(lengths))
   return windows[idx]


def slice_returns(returns: pd.DataFrame, window: Tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
   start, end = window
   return returns.loc[(returns.index >= start) & (returns.index <= end)]


def regime_correlation_and_volatility(returns: pd.DataFrame, window: Tuple[pd.Timestamp, pd.Timestamp]):
   slice_r = slice_returns(returns, window)
   corr = slice_r.corr()
   vol = slice_r.std()
   return corr, vol


def plot_heatmap(matrix: pd.DataFrame, title: str, fname: str):
   """
   Standard Heatmap. No dendrograms. No annotations.
   """
   fig, ax = plt.subplots(figsize=(10, 8))
  
   # annot=False removes numbers for clarity
   # cmap="coolwarm" goes from blue (low) to red (high) which provides good visual contrast
   sns.heatmap(matrix, annot=False, cmap="coolwarm", vmin=0.0, vmax=1.0, ax=ax)
  
   ax.set_title(title)
   fig.tight_layout()
   path = os.path.join(FIGURES_DIR, fname)
   fig.savefig(path, dpi=300, bbox_inches="tight")
   plt.close(fig)
   print(f"Saved: {path}")


def main():
   returns = load_returns(RETURNS_PATH)

   market_idx = build_market_index(returns)
   market_idx.to_csv(os.path.join(OUTPUTS_DIR, "market_index.csv"))
  
   # 1.Compute downside vol
   rolling_downside = compute_downside_volatility(returns, window=VOL_WINDOW)
   rolling_downside.to_csv(os.path.join(OUTPUTS_DIR, "market_downside_vol.csv"))
   # 2. Determine Threshold (Dynamic / Expanding Window)
   # min_periods=365 ensures we have 1 year of history before classifying stress.
   dynamic_threshold = rolling_downside.expanding(min_periods=365).quantile(VOL_PERCENTILE)
   
   # Create Boolean Mask: Is today's volatility higher than the historical 85th percentile?
   is_stress_mask = rolling_downside > dynamic_threshold
   
   print(f"Computed dynamic volatility thresholds (Expanding Window).")

   # 3.Detect Stress Periods
   stress_windows = detect_stress_periods(
       stress_mask=is_stress_mask,
       min_len=2
   )

   # Save Regimes
   regimes_df = []
   for start, end in stress_windows:
       regimes_df.append({"kind": "stress", "start": start, "end": end})

   # Normal periods (gaps)
   all_dates = rolling_downside.dropna().index
   stress_intervals = [(s, e) for s, e in stress_windows]
  
   if stress_intervals:
       prev_end = all_dates[0] - pd.Timedelta(days=1)
       for s, e in stress_intervals:
           if (s - prev_end).days > 1:
               normal_start = prev_end + pd.Timedelta(days=1)
               normal_end = s - pd.Timedelta(days=1)
               if normal_start <= normal_end:
                    regimes_df.append({"kind": "normal", "start": normal_start, "end": normal_end})
           prev_end = e
       if prev_end < all_dates[-1]:
           regimes_df.append({"kind": "normal", "start": prev_end + pd.Timedelta(days=1), "end": all_dates[-1]})
   else:
       if not all_dates.empty:
           regimes_df.append({"kind": "normal", "start": all_dates[0], "end": all_dates[-1]})

   if regimes_df:
       pd.DataFrame(regimes_df).to_csv(os.path.join(OUTPUTS_DIR, "detected_regimes.csv"), index=False)
       print(f"Detected {len(stress_windows)} stress windows.")

   if not stress_windows:
       print("No stress windows detected. Exiting.")
       return

   # 4. Pick representative periods
   stress_window = pick_representative_period(stress_windows)
   stress_len = (stress_window[1] - stress_window[0]).days + 1

   # Find Normal window
   valid_dates = rolling_downside.dropna().index
   candidate = None
  
   for center in valid_dates:
       start = center - pd.Timedelta(days=stress_len // 2)
       end = start + pd.Timedelta(days=stress_len - 1)
      
       if start >= valid_dates[0] and end <= valid_dates[-1]:
           # Check if ANY day in this window was classified as stress
           # If not (all False), then it's a valid Normal window
           if not is_stress_mask.loc[start:end].any():
               candidate = (start, end)
               break
  
   if candidate is None:
       candidate = (valid_dates[0], valid_dates[0] + pd.Timedelta(days=stress_len - 1))
   normal_window = candidate

   print(f"Stress Window: {stress_window}")
   print(f"Normal Window: {normal_window}")

   # Compute stats
   corr_stress, vol_stress = regime_correlation_and_volatility(returns, stress_window)
   corr_normal, vol_normal = regime_correlation_and_volatility(returns, normal_window)

   corr_stress.to_csv(os.path.join(OUTPUTS_DIR, "corr_stress.csv"))
   corr_normal.to_csv(os.path.join(OUTPUTS_DIR, "corr_normal.csv"))
  
   # Standard Heatmaps
   plot_heatmap(corr_normal, "Correlation - Normal", "corr_normal_heatmap.png")
   plot_heatmap(corr_stress, "Correlation - Stress", "corr_stress_heatmap.png")
  
   # Diff Heatmap
   corr_diff = corr_stress - corr_normal
   fig, ax = plt.subplots(figsize=(10, 8))
   sns.heatmap(corr_diff, annot=False, cmap="coolwarm", center=0, vmin=-0.5, vmax=0.5, ax=ax)
   ax.set_title("Correlation Difference (Stress - Normal)")
   fig.tight_layout()
   fig.savefig(os.path.join(FIGURES_DIR, "corr_diff_heatmap.png"), dpi=300, bbox_inches="tight")
   plt.close(fig)

   print("Analysis complete.")


def load_regime_outputs():
   """
   Load precomputed regime outputs for main.py metrics
   """
   OUTPUTS_DIR = "results/outputs"

   #1. Load Correlations
   corr_normal = pd.read_csv(os.path.join(OUTPUTS_DIR, "corr_normal.csv"), index_col=0)
   corr_stress = pd.read_csv(os.path.join(OUTPUTS_DIR, "corr_stress.csv"), index_col=0)

   #2. Load Volatility 
   vol_normal = pd.read_csv(os.path.join(OUTPUTS_DIR, "vol_normal.csv"), index_col=0)
   if vol_normal.shape[1] == 1:
       vol_normal = vol_normal.iloc[:, 0]

   vol_stress = pd.read_csv(os.path.join(OUTPUTS_DIR, "vol_stress.csv"), index_col=0)
   if vol_stress.shape[1] == 1:
       vol_stress = vol_stress.iloc[:, 0]

   #3. Cluster labels
   cluster_labels_path = os.path.join(OUTPUTS_DIR, "cluster_labels.csv")
   if os.path.exists(cluster_labels_path):
       cluster_labels = pd.read_csv(cluster_labels_path, index_col=0).iloc[:, 0]
   else:
       cluster_labels = pd.Series(dtype=float)

   return corr_normal, corr_stress, vol_normal, vol_stress, cluster_labels


if __name__ == "__main__":
   main()