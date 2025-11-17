"""
Merge all raw CSV datasets into a single clean CSV file.

The clean dataset will contain only the dates and 
one column per cryptocurrency with daily price.
"""
import pandas as pd
import os

coins = [
    "bitcoin",
    "ethereum",
    "binancecoin",
    "solana",
    "cardano",
    "xrp",
    "polkadot",
    "dogecoin",
    "avalanche",
    "litecoin",
]

data_folder = "data/raw"
all_dfs = []

for coin in coins:
    file_path = os.path.join(data_folder, f"{coin}.csv")
    df = pd.read_csv(file_path)
    df = df[["snapped_at", "price"]]
    df["snapped_at"] = pd.to_datetime(df["snapped_at"])
    df.rename(columns={"price": coin}, inplace=True) # rename price column to coin name
    df.set_index("snapped_at", inplace=True)
    all_dfs.append(df)

# merge all coins, keeping only dates present in every dataset
merged_df = pd.concat(all_dfs, axis=1, join="inner")
merged_df.to_csv(os.path.join(data_folder, "clean_prices.csv"))

print("Merged and cleaned datasets into clean_prices.csv")
