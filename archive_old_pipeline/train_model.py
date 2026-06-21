import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
import joblib
from data_pipeline import transform_traffic_data, compute_disruption_index

def assign_urgency_tiers(series: pd.Series) -> pd.Series:
    bins = [-np.inf, 25, 50, 75, np.inf]
    labels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    return pd.cut(series, bins=bins, labels=labels, right=False)

def train_predictive_model(csv_path: str):
    print("Loading raw dataset...")
    df_raw = pd.read_csv(csv_path)
    
    print("Executing offline preprocessing pipelines...")
    df_transformed = transform_traffic_data(df_raw)
    df_final = compute_disruption_index(df_transformed)
    
    print("Performing strict chronological split (80/20)...")
    df_final = df_final.sort_values('validation_timestamp', ascending=True).reset_index(drop=True)
    
    split_index = int(len(df_final) * 0.8)
    train_df = df_final.iloc[:split_index].copy()
    test_df = df_final.iloc[split_index:].copy()
    
    print(f"Training Set Timeline:   {train_df['validation_timestamp'].min()} to {train_df['validation_timestamp'].max()}")
    print(f"Validation Set Timeline: {test_df['validation_timestamp'].min()} to {test_df['validation_timestamp'].max()}")
    
    # Target distribution diagnostics
    print(f"Target Check -> Max Train TDI: {train_df['calculated_tdi'].max():.2f} | Max Test TDI: {test_df['calculated_tdi'].max():.2f}")
    
    features = [
        'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 
        'pcu_weight', 'active_concurrent_violations', 'rolling_pcu_load',
        'junction_name', 'police_station', 'device_id'
    ]
    categorical_features = ['junction_name', 'police_station', 'device_id']
    target = 'calculated_tdi'
    
    for col in categorical_features:
        train_df[col] = train_df[col].fillna('UNKNOWN').astype('category')
        test_df[col] = pd.Categorical(test_df[col].fillna('UNKNOWN'), categories=train_df[col].cat.categories)
        
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    print("\nInitializing LightGBM Regressor (Fixed Params)...")
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training predictive model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        categorical_feature=categorical_features
    )
        
    print("\n--- Model Evaluation metrics ---")
    y_pred = model.predict(X_test)
    print(f"Test MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    
    print("\n--- Operational Tiering Performance ---")
    y_test_tiers = assign_urgency_tiers(y_test)
    y_pred_tiers = assign_urgency_tiers(pd.Series(y_pred, index=y_test.index))
    
    true_critical = (y_test_tiers == 'CRITICAL')
    pred_critical = (y_pred_tiers == 'CRITICAL')
    
    print("Classification Report for CRITICAL anomalies:")
    print(classification_report(true_critical, pred_critical, labels=[False, True], target_names=['Non-Critical', 'Critical'], zero_division=0))
    
    print("\nSerializing trained model...")
    joblib.dump(model, 'traffic_model.pkl')
    print("Model successfully saved as 'traffic_model.pkl'")

if __name__ == "__main__":
    # Make sure this matches your local 300k filename exactly!
    train_predictive_model("dataset1.csv")