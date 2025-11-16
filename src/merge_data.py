import pandas as pd
import os

coins = ["bitcoin", "ethereum", "binancecoin", "solana", "cardano", 
         "xrp", "polkadot", "dogecoin", "avalanche", "litecoin"]

data_folder = "data"
all_dfs = []

for coin in coins:
    file_path = os.path.join(data_folder, f"{coin}.csv")
    df = pd.read_csv(file_path)
    df = df[["snapped_at", "price"]]
    df["snapped_at"] = pd.to_datetime(df["snapped_at"])
    df.rename(columns={"price": coin}, inplace=True)
    df.set_index("snapped_at", inplace=True)
    all_dfs.append(df)

merged_df = pd.concat(all_dfs, axis=1, join="inner")
merged_df.to_csv(os.path.join(data_folder, "clean_prices.csv"))

print("Merged and cleaned datasets into clean_prices.csv")