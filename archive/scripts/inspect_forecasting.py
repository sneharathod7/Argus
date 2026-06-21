import pandas as pd

df = pd.read_parquet("forecasting_dataset.parquet")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nTarget Distribution:")
print(df["severity_target"].value_counts(normalize=True) * 100)

print("\nMissing Values:")
print(df.isna().sum().sort_values(ascending=False).head(20))