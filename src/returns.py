"""
Compute daily returns for each coin and save as returns.csv
"""
import pandas as pd
import os

data_folder = "data/clean"
merged_file = os.path.join(data_folder, "clean_prices.csv")
returns_file = os.path.join(data_folder, "returns.csv")

# read merged prices
df = pd.read_csv(merged_file, index_col=0, parse_dates=True)

# calculate daily returns in percentage change from one day to next
returns = df.pct_change().dropna() 

# save
returns.to_csv(returns_file)

print("Daily returns saved to returns.csv")
