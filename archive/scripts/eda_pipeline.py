import os
import ast
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_json_column(df, col_name):
    def safe_parse(val):
        if pd.isna(val):
            return []
        try:
            return json.loads(val)
        except:
            try:
                return ast.literal_eval(val)
            except:
                return [val]
    return df[col_name].apply(safe_parse)

def run_eda(file_path, output_dir):
    print("Starting EDA Pipeline...")
    create_directory(output_dir)
    
    # Load dataset
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(df)} records.")
    
    # Convert datetimes
    print("Converting datetimes...")
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    
    # Feature Engineering
    print("Engineering temporal features...")
    df['hour'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.day_name()
    df['month'] = df['created_datetime'].dt.month_name()
    df['year_month'] = df['created_datetime'].dt.to_period('M')
    
    sns.set_theme(style="whitegrid")
    
    report_content = ["# Traffic Intelligence Executive Summary\n"]
    report_content.append(f"**Total Records Analyzed:** {len(df):,}\n")

    print("1. Plotting Top 20 Hotspot Junctions...")
    plt.figure(figsize=(12, 8))
    # Fill missing junction names with 'Unknown'
    df['junction_name'] = df['junction_name'].fillna('Unknown')
    top_junctions = df['junction_name'].value_counts().head(20)
    sns.barplot(y=top_junctions.index, x=top_junctions.values, palette='viridis')
    plt.title('Top 20 Hotspot Junctions by Violation Volume')
    plt.xlabel('Number of Violations')
    plt.ylabel('Junction Name')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_top_20_junctions.png'))
    plt.close()
    
    report_content.append("## 1. Top Junctions")
    report_content.append(f"The top junction is **{top_junctions.index[0]}** with {top_junctions.values[0]:,} violations.\n")

    print("2. Plotting Top Police Stations...")
    plt.figure(figsize=(12, 10))
    df['police_station'] = df['police_station'].fillna('Unknown')
    top_ps = df['police_station'].value_counts().head(20)
    sns.barplot(y=top_ps.index, x=top_ps.values, palette='magma')
    plt.title('Top Police Stations by Violation Volume')
    plt.xlabel('Number of Violations')
    plt.ylabel('Police Station')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_top_police_stations.png'))
    plt.close()
    
    print("3. Plotting Hourly Violation Distribution...")
    plt.figure(figsize=(10, 6))
    hourly_dist = df['hour'].value_counts().sort_index()
    sns.barplot(x=hourly_dist.index, y=hourly_dist.values, color='steelblue')
    plt.title('Hourly Violation Distribution')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Volume')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_hourly_distribution.png'))
    plt.close()

    print("4. Plotting Day-of-Week Distribution...")
    plt.figure(figsize=(10, 6))
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.countplot(data=df, x='day_of_week', order=order, palette='crest')
    plt.title('Day-of-Week Violation Distribution')
    plt.xlabel('Day of Week')
    plt.ylabel('Volume')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_dow_distribution.png'))
    plt.close()

    print("5. Plotting Monthly Violation Distribution...")
    plt.figure(figsize=(12, 6))
    monthly_dist = df['year_month'].value_counts().sort_index()
    monthly_dist.index = monthly_dist.index.astype(str)
    sns.barplot(x=monthly_dist.index, y=monthly_dist.values, palette='rocket')
    plt.title('Monthly Violation Distribution')
    plt.xlabel('Month')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_monthly_distribution.png'))
    plt.close()

    print("6. Plotting Vehicle Type Distribution...")
    plt.figure(figsize=(12, 6))
    veh_type = df['vehicle_type'].value_counts().head(15)
    sns.barplot(y=veh_type.index, x=veh_type.values, palette='cubehelix')
    plt.title('Top Vehicle Types Involved in Violations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_vehicle_type_distribution.png'))
    plt.close()

    print("7. Plotting Vehicle Type vs Hotspot...")
    top_5_veh = df['vehicle_type'].value_counts().nlargest(5).index
    top_10_junc = df['junction_name'].value_counts().nlargest(10).index
    subset = df[(df['vehicle_type'].isin(top_5_veh)) & (df['junction_name'].isin(top_10_junc))]
    
    if not subset.empty:
        plt.figure(figsize=(14, 8))
        pivot_junc_veh = pd.crosstab(subset['junction_name'], subset['vehicle_type'])
        pivot_junc_veh.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
        plt.title('Vehicle Type vs Top 10 Hotspot Junctions')
        plt.xlabel('Junction Name')
        plt.ylabel('Volume')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Vehicle Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '07_vehicle_vs_hotspot.png'))
        plt.close()

    print("8 & 9. Parsing Violation Types and Combinations...")
    df['parsed_violations'] = parse_json_column(df, 'violation_type')
    all_violations = [item for sublist in df['parsed_violations'] for item in sublist]
    viol_counts = pd.Series(all_violations).value_counts().head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(y=viol_counts.index, x=viol_counts.values, palette='flare')
    plt.title('Top Violation Types')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_violation_types.png'))
    plt.close()
    
    # Combinations
    df['violation_combo'] = df['parsed_violations'].apply(lambda x: ' + '.join(sorted(x)))
    combo_counts = df[df['parsed_violations'].map(len) > 1]['violation_combo'].value_counts().head(10)
    
    if not combo_counts.empty:
        plt.figure(figsize=(12, 8))
        sns.barplot(y=combo_counts.index, x=combo_counts.values, palette='mako')
        plt.title('Most Frequent Violation Combinations (2+ offenses)')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '09_violation_combinations.png'))
        plt.close()

    print("10. Plotting Validation Status...")
    plt.figure(figsize=(8, 6))
    val_status = df['validation_status'].fillna('Missing/Unknown').value_counts()
    sns.barplot(x=val_status.index, y=val_status.values, palette='Set2')
    plt.title('Validation Status Distribution')
    plt.xlabel('Status')
    plt.ylabel('Volume')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_validation_status.png'))
    plt.close()

    print("11 & 12. Plotting GPS Heatmap...")
    plt.figure(figsize=(12, 10))
    # Filter valid coordinates for Bangalore roughly to avoid extreme outliers messing up the plot
    geo_df = df[(df['latitude'] > 12.0) & (df['latitude'] < 14.0) & 
                (df['longitude'] > 77.0) & (df['longitude'] < 78.5)].copy()
    if not geo_df.empty:
        plt.hexbin(geo_df['longitude'], geo_df['latitude'], gridsize=100, cmap='inferno', mincnt=1)
        plt.colorbar(label='Number of Violations')
        plt.title('Spatial Heatmap of Violations (Bengaluru)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '11_12_gps_heatmap.png'))
    plt.close()

    print("13, 14, 15. Analyzing Persistent Hotspots Across Months...")
    # Find junctions that appear in top 20 across multiple months
    junc_month = df.groupby(['year_month', 'junction_name']).size().reset_index(name='count')
    # Filter out 'No Junction' or 'Unknown' for true hotspot analysis if desired, but we keep them for now
    valid_juncs = junc_month[~junc_month['junction_name'].isin(['Unknown', 'No Junction'])]
    
    top_per_month = valid_juncs.sort_values(['year_month', 'count'], ascending=[True, False]).groupby('year_month').head(10)
    persistent_juncs = top_per_month['junction_name'].value_counts()
    persistent = persistent_juncs[persistent_juncs > 1]
    
    if not persistent.empty:
        plt.figure(figsize=(12, 8))
        sns.barplot(y=persistent.index[:15], x=persistent.values[:15], palette='Spectral')
        plt.title('Persistent Hotspots (Appeared in Top 10 across multiple months)')
        plt.xlabel('Number of Months in Top 10')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '13_14_15_persistent_hotspots.png'))
    plt.close()
    
    report_content.append("## Key Insights\n")
    report_content.append(f"- **Peak Time Trends**: A review of the hourly distribution (plot 03) reveals the active enforcement periods.\n")
    
    if not persistent.empty:
        report_content.append(f"- **Persistent Hotspots**: {len(persistent)} junctions frequently appeared in the top 10 most violated hotspots across different months, pointing to persistent geometric or structural traffic issues.\n")
    if not combo_counts.empty:
        report_content.append(f"- **Severe Combinations**: The most frequent severe violation combination is '{combo_counts.index[0]}' with {combo_counts.values[0]:,} occurrences.\n")
    
    report_content.append("- **Validation Status**: The target validation column exhibits significant missing values (`Missing/Unknown`), which requires operational intervention or semi-supervised approaches to extract full value.\n")
    
    report_content.append("\n## Deliverables\n")
    report_content.append("All requested charts have been saved to the output directory as PNG files. Please review the `eda_outputs` directory for all visual insights.")
    
    with open(os.path.join(output_dir, 'executive_summary.md'), 'w') as f:
        f.write('\n'.join(report_content))
        
    print(f"EDA pipeline complete. Outputs saved in '{output_dir}'.")

if __name__ == "__main__":
    run_eda("dataset1.csv", "eda_outputs")
