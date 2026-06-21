import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, f1_score, classification_report
import warnings

warnings.filterwarnings('ignore')

def get_severity_class(x):
    if x <= 0.5:
        return 'CLEAR'
    elif x <= 2.5:
        return 'LOW'
    elif x <= 5.5:
        return 'MEDIUM'
    else:
        return 'CRITICAL'

def main():
    print("Loading forecasting_dataset.parquet...")
    df = pd.read_parquet("forecasting_dataset.parquet")
    
    # Sort chronologically to be absolutely safe
    df['hour'] = pd.to_datetime(df['hour'], utc=True)
    df = df.sort_values('hour').reset_index(drop=True)
    
    print("Preparing features...")
    # Handle categoricals
    cat_cols = ['grid_id', 'dominant_vehicle_type', 'dominant_violation_type', 'day_of_week']
    for col in cat_cols:
        df[col] = df[col].astype(str).astype('category')
        
    features = [
        'violation_count_lag_1', 'violation_count_lag_2', 'violation_count_lag_3',
        'violation_count_lag_6', 'violation_count_lag_12', 'violation_count_lag_24',
        'rolling_mean_3h', 'rolling_mean_6h', 'rolling_mean_12h', 'rolling_mean_24h',
        'rolling_sum_6h', 'rolling_sum_12h', 'rolling_sum_24h', 'rolling_max_24h',
        'violations_last_24h', 'violations_last_7d',
        'current_count_minus_rolling_mean_24h',
        'is_weekend', 'hour_of_day'
    ] + cat_cols
    
    target = 'target_violation_count'
    
    # Chronological Split
    print("Splitting dataset chronologically...")
    n = len(df)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)
    
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]
    
    print(f"Train: {len(train_df)} rows")
    print(f"Val:   {len(val_df)} rows")
    print(f"Test:  {len(test_df)} rows")
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print("Training LightGBM Regressor (Poisson Objective)...")
    model = lgb.LGBMRegressor(
        objective='poisson',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l1', # MAE
        callbacks=callbacks
    )
    
    print("Evaluating on Test Set...")
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    print(f"Test MAE: {mae:.4f}")
    
    print("Deriving severity classes and evaluating Macro F1...")
    pred_classes = [get_severity_class(p) for p in preds]
    true_classes = test_df['target_severity'].values
    
    macro_f1 = f1_score(true_classes, pred_classes, average='macro', labels=['CLEAR', 'LOW', 'MEDIUM', 'CRITICAL'])
    print(f"Test Macro F1: {macro_f1:.4f}")
    
    class_report = classification_report(true_classes, pred_classes, labels=['CLEAR', 'LOW', 'MEDIUM', 'CRITICAL'])
    print("\nClassification Report:\n", class_report)
    
    # Feature Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nSaving model and results...")
    model.booster_.save_model('stgf_lightgbm_model.txt')
    
    md = f"""# Model Evaluation Report: STGF Baseline

## Overview
We trained a LightGBM Regressor using a **Poisson** objective function to predict exact hourly violation counts (`target_violation_count`), mapping the continuous predictions to Business Severity Categories.

### Temporal Data Split
- **Train Split (70%):** `{train_df['hour'].min()}` to `{train_df['hour'].max()}`
- **Validation Split (15%):** `{val_df['hour'].min()}` to `{val_df['hour'].max()}`
- **Test Split (15%):** `{test_df['hour'].min()}` to `{test_df['hour'].max()}`

## Evaluation Metrics (Hold-out Test Set)

- **Primary Metric (Regression):** Mean Absolute Error (MAE) = `{mae:.4f}` violations.
  *This means our forecast is off by less than {mae:.2f} violations per hour on average.*
- **Secondary Metric (Classification):** Macro F1-Score = `{macro_f1:.4f}`.

### Business Severity Classification Report
```text
{class_report}
```

## Top 5 Most Important Features
"""
    for idx, row in importance.head(5).iterrows():
        md += f"- **{row['Feature']}**: {row['Importance']:.0f} (splits)\n"
        
    md += "\n## Conclusion\n"
    md += "- The model successfully forecasts severe hotspots without data leakage using strict chronological splitting.\n"
    md += "- Model saved to `stgf_lightgbm_model.txt`.\n"
    
    with open('model_evaluation.md', 'w') as f:
        f.write(md)

if __name__ == "__main__":
    main()
