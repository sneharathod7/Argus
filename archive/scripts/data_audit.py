import pandas as pd
import json
import sys

def analyze_dataset(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Shape: {df.shape}")
    
    audit = {}
    audit["shape"] = df.shape
    audit["columns"] = df.columns.tolist()
    
    # 1. Missing values
    missing = df.isnull().sum().to_dict()
    audit["missing_values"] = {k: int(v) for k, v in missing.items() if v > 0}
    
    # Data Types
    audit["dtypes"] = {k: str(v) for k, v in df.dtypes.items()}
    
    # Column details
    col_details = {}
    for col in df.columns:
        col_info = {}
        col_info["unique_count"] = df[col].nunique()
        col_info["missing_count"] = int(df[col].isnull().sum())
        col_info["dtype"] = str(df[col].dtype)
        # Get some sample values
        col_info["sample_values"] = [str(x) for x in df[col].dropna().unique()[:5]]
        
        # If numeric, get min/max
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min())
            col_info["max"] = float(df[col].max())
            col_info["mean"] = float(df[col].mean())
            
        col_details[col] = col_info
        
    audit["column_details"] = col_details
    
    # Duplicates
    audit["duplicate_rows"] = int(df.duplicated().sum())
    
    with open('audit_results.json', 'w') as f:
        json.dump(audit, f, indent=4)
        
    print("Audit saved to audit_results.json")

if __name__ == "__main__":
    analyze_dataset("dataset1.csv")
