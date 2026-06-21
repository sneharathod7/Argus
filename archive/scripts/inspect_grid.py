import pandas as pd

df = pd.read_parquet("grid_hourly_table.parquet")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nHead:")
print(df.head())

print("\nViolation Count Stats:")
print(df["violation_count"].describe())

print("\nTop 20 violation counts:")
print(df["violation_count"].value_counts().head(20))