""" import pandas as pd

from data_preprocessing import load_rossmann_data, preprocess_data

# Step 1: Load raw sales data
raw_sales = pd.read_csv("Data/train.csv")
print(f"Step 1 - Raw sales data shape: {raw_sales.shape}")

# Step 2: Filter out closed stores (Sales = 0)
filtered_sales = raw_sales[raw_sales['Sales'] > 0]
print(f"Step 2 - After removing closed stores: {filtered_sales.shape}")

# Step 3: Load store data and merge
store_data = pd.read_csv("Data/store.csv")
print(f"showoff - Raw store data shape: {store_data.shape}")
merged_df = pd.merge(filtered_sales, store_data, on="Store", how="left")
print(f"Step 3 - After merging with store data: {merged_df.shape}")

# Step 4: Filter only first 5 store IDs
store_ids = merged_df['Store'].unique()[:5]
df_subset = merged_df[merged_df['Store'].isin(store_ids)]
print(f"Step 4 - Subset with 5 stores: {df_subset.shape}")

# Step 5: Preprocess the subset
processed_df = preprocess_data(df_subset)
print(f"Step 5 - After preprocessing: {processed_df.shape}")

# Step 6: Compare rows
rows_before = df_subset.shape[0]
rows_after = processed_df.shape[0]
print(f"\nRows before preprocessing: {rows_before}")
print(f"Rows after preprocessing:  {rows_after}")
print(f"Rows reduced:              {rows_before - rows_after}")
print(f"% reduction:               {((rows_before - rows_after) / rows_before) * 100:.2f}%")
 """