"""
Phase 4: Two-Stage Zero-Inflated Model + Ensemble
====================================================
Implements the elite modeling pipeline:

  Stage A: Hotspot Occurrence Model (binary: will violations happen?)
           - LightGBM with Focal Loss
           
  Stage B: Conditional Intensity Model (regression: how many, given active?)
           - LightGBM (Tweedie) + CatBoost (Tweedie) ensemble
           - Weighted blend (OOF stacking if feasible, else weighted average)

  Risk Score: Composite ranking for enforcement dispatch

  Bayesian Priors: Grid-level historical rates computed on training data only

  Hyperparameter Optimization: Optuna

Input:  elite_forecasting_dataset.parquet (from Phase 3)
Output: Models (.txt/.cbm) + evaluation report
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error, f1_score, precision_score, recall_score,
    average_precision_score, log_loss, classification_report,
    mean_squared_error
)
from sklearn.model_selection import TimeSeriesSplit
import optuna
import warnings
import time
import os
import json

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────
# Focal Loss for LightGBM (custom objective)
# ─────────────────────────────────────────────────────────
def focal_loss_objective(y_pred, dtrain, gamma=2.0, alpha=0.25):
    """Focal Loss: down-weights easy negatives, focuses on hard examples."""
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
    
    # Gradient
    grad = alpha * y_true * (1 - p)**gamma * (gamma * p * np.log(p + 1e-9) + p - 1) + \
           (1 - alpha) * (1 - y_true) * p**gamma * (-gamma * (1 - p) * np.log(1 - p + 1e-9) + p)
    
    # Hessian (approximation)
    hess = alpha * y_true * (1 - p)**gamma * p * (
        gamma * (gamma * p * np.log(p + 1e-9) + 2 * p - 1) + p
    ) + (1 - alpha) * (1 - y_true) * p**gamma * (1 - p) * (
        gamma * (-gamma * (1 - p) * np.log(1 - p + 1e-9) + 2 * (1 - p) - 1) + (1 - p)
    )
    hess = np.maximum(hess, 1e-6)  # ensure positive
    
    return grad, hess


def focal_loss_eval(y_pred, dtrain):
    """Evaluation metric: binary log loss (for early stopping)."""
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return 'focal_logloss', loss, False  # lower is better


def main():
    start_time = time.time()
    print("=" * 70)
    print("PHASE 4: TWO-STAGE MODEL + ENSEMBLE")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────
    # Load data
    # ─────────────────────────────────────────────────────────
    print("\n[1/8] Loading elite_forecasting_dataset.parquet...")
    df = pd.read_parquet("elite_forecasting_dataset.parquet")
    df['hour'] = pd.to_datetime(df['hour'], utc=True)
    df = df.sort_values('hour').reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ─────────────────────────────────────────────────────────
    # Define features
    # ─────────────────────────────────────────────────────────
    print("\n[2/8] Preparing features...")

    cat_cols = ['grid_id', 'dominant_vehicle_type', 'dominant_violation_type', 'day_of_week', 'police_station']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).astype('category')

    # All numeric + categorical features (exclude targets and identifiers)
    exclude_cols = {
        'hour', 'target_violation_count_1h', 'target_violation_count_2h',
        'target_violation_count_3h', 'target_is_active_1h', 'target_is_active_2h',
        'target_is_active_3h', 'target_severity_1h',
        'grid_lat', 'grid_lng',  # coordinates used for neighbor computation, not features
    }

    # Build feature list dynamically
    features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'int32', 'float32', 'bool', 'category', 'int8', 'uint8']]
    
    # Ensure categorical columns are included
    for c in cat_cols:
        if c in df.columns and c not in features:
            features.append(c)

    print(f"  Total features: {len(features)}")
    print(f"  Categorical: {[c for c in features if c in cat_cols]}")

    # ─────────────────────────────────────────────────────────
    # Chronological Split
    # ─────────────────────────────────────────────────────────
    print("\n[3/8] Chronological train/val/test split...")
    n = len(df)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()

    print(f"  Train: {len(train_df):,} rows ({train_df['hour'].min()} to {train_df['hour'].max()})")
    print(f"  Val:   {len(val_df):,} rows ({val_df['hour'].min()} to {val_df['hour'].max()})")
    print(f"  Test:  {len(test_df):,} rows ({test_df['hour'].min()} to {test_df['hour'].max()})")

    # ─────────────────────────────────────────────────────────
    # Bayesian Historical Prior (computed on train only)
    # ─────────────────────────────────────────────────────────
    print("\n[4/8] Computing Bayesian historical priors...")

    # Grid-level historical hotspot rate with Bayesian smoothing
    alpha_smooth = 10  # smoothing parameter
    train_df['_is_active'] = (train_df['violation_count'] > 0).astype(int)
    train_df['_is_critical'] = (train_df['violation_count'] >= 6).astype(int)

    global_active_rate = train_df['_is_active'].mean()
    global_critical_rate = train_df['_is_critical'].mean()

    grid_stats = train_df.groupby('grid_id').agg(
        n_hours=('_is_active', 'count'),
        n_active=('_is_active', 'sum'),
        n_critical=('_is_critical', 'sum'),
        mean_violations=('violation_count', 'mean'),
        peak_hour=('hour_of_day', lambda x: x[train_df.loc[x.index, 'violation_count'] > 0].mode().iloc[0] if (train_df.loc[x.index, 'violation_count'] > 0).any() else 12),
    ).reset_index()

    # Bayesian smoothed rates
    grid_stats['bayesian_active_rate'] = (
        (grid_stats['n_active'] + alpha_smooth * global_active_rate) /
        (grid_stats['n_hours'] + alpha_smooth)
    )
    grid_stats['bayesian_critical_rate'] = (
        (grid_stats['n_critical'] + alpha_smooth * global_critical_rate) /
        (grid_stats['n_hours'] + alpha_smooth)
    )

    # Same-hour-DOW historical average
    grid_hour_dow_avg = train_df.groupby(['grid_id', 'hour_of_day', 'day_of_week_num'])['violation_count'].mean().reset_index()
    grid_hour_dow_avg.rename(columns={'violation_count': 'same_hour_dow_hist_avg'}, inplace=True)

    # Response time aggregate per grid (leakage-safe: train only)
    grid_response = train_df.groupby('grid_id')['avg_response_time'].mean().reset_index()
    grid_response.rename(columns={'avg_response_time': 'grid_avg_response_time'}, inplace=True)

    # Merge priors into all splits
    prior_cols = ['grid_id', 'bayesian_active_rate', 'bayesian_critical_rate', 'mean_violations', 'peak_hour']
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=[c for c in prior_cols[1:] if c in split_df.columns], inplace=True, errors='ignore')

    train_df = train_df.merge(grid_stats[prior_cols], on='grid_id', how='left')
    val_df = val_df.merge(grid_stats[prior_cols], on='grid_id', how='left')
    test_df = test_df.merge(grid_stats[prior_cols], on='grid_id', how='left')

    # Merge same-hour-dow averages
    for split_df in [train_df, val_df, test_df]:
        if 'same_hour_dow_hist_avg' in split_df.columns:
            split_df.drop(columns=['same_hour_dow_hist_avg'], inplace=True)
    train_df = train_df.merge(grid_hour_dow_avg, on=['grid_id', 'hour_of_day', 'day_of_week_num'], how='left')
    val_df = val_df.merge(grid_hour_dow_avg, on=['grid_id', 'hour_of_day', 'day_of_week_num'], how='left')
    test_df = test_df.merge(grid_hour_dow_avg, on=['grid_id', 'hour_of_day', 'day_of_week_num'], how='left')

    # Merge response time
    for split_df in [train_df, val_df, test_df]:
        if 'grid_avg_response_time' in split_df.columns:
            split_df.drop(columns=['grid_avg_response_time'], inplace=True)
    train_df = train_df.merge(grid_response, on='grid_id', how='left')
    val_df = val_df.merge(grid_response, on='grid_id', how='left')
    test_df = test_df.merge(grid_response, on='grid_id', how='left')

    # Fill unseen grids with global averages
    for split_df in [train_df, val_df, test_df]:
        split_df['bayesian_active_rate'] = split_df['bayesian_active_rate'].fillna(global_active_rate)
        split_df['bayesian_critical_rate'] = split_df['bayesian_critical_rate'].fillna(global_critical_rate)
        split_df['mean_violations'] = split_df['mean_violations'].fillna(0)
        split_df['peak_hour'] = split_df['peak_hour'].fillna(12)
        split_df['same_hour_dow_hist_avg'] = split_df['same_hour_dow_hist_avg'].fillna(0)
        split_df['grid_avg_response_time'] = split_df['grid_avg_response_time'].fillna(0)

    # Is current hour the grid's historical peak?
    for split_df in [train_df, val_df, test_df]:
        split_df['is_grid_peak_hour'] = (split_df['hour_of_day'] == split_df['peak_hour']).astype(int)

    # Update feature list with new Bayesian features
    bayesian_features = [
        'bayesian_active_rate', 'bayesian_critical_rate', 'mean_violations',
        'same_hour_dow_hist_avg', 'is_grid_peak_hour', 'grid_avg_response_time'
    ]
    features_full = features + bayesian_features

    # Clean up temp columns
    train_df.drop(columns=['_is_active', '_is_critical', 'peak_hour'], inplace=True, errors='ignore')

    print(f"  Bayesian features added: {bayesian_features}")
    print(f"  Total features for modeling: {len(features_full)}")

    # ─────────────────────────────────────────────────────────
    # STAGE A: Hotspot Occurrence Model
    # ─────────────────────────────────────────────────────────
    print("\n[5/8] STAGE A: Training Hotspot Occurrence Model...")

    target_a = 'target_is_active_1h'

    X_train_a = train_df[features_full]
    y_train_a = train_df[target_a]
    X_val_a = val_df[features_full]
    y_val_a = val_df[target_a]
    X_test_a = test_df[features_full]
    y_test_a = test_df[target_a]

    pos_rate = y_train_a.mean()
    print(f"  Positive rate (train): {pos_rate:.4f}")

    # Quick Optuna tuning for Stage A
    print("  Running Optuna hyperparameter search (50 trials)...")

    def objective_a(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
        }
        
        dtrain = lgb.Dataset(X_train_a, y_train_a, categorical_feature=[c for c in cat_cols if c in features_full])
        dval = lgb.Dataset(X_val_a, y_val_a, reference=dtrain)
        
        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        
        val_preds = model.predict(X_val_a)
        return average_precision_score(y_val_a, val_preds)

    study_a = optuna.create_study(direction='maximize')
    study_a.optimize(objective_a, n_trials=50, show_progress_bar=False)

    best_params_a = study_a.best_params
    print(f"  Best PR-AUC: {study_a.best_value:.4f}")
    print(f"  Best params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best_params_a.items()})}")

    # Train final Stage A model with best params
    final_params_a = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'n_jobs': -1,
        'is_unbalance': True,
        **best_params_a
    }

    dtrain_a = lgb.Dataset(X_train_a, y_train_a, categorical_feature=[c for c in cat_cols if c in features_full])
    dval_a = lgb.Dataset(X_val_a, y_val_a, reference=dtrain_a)

    model_a = lgb.train(
        final_params_a, dtrain_a,
        num_boost_round=1000,
        valid_sets=[dval_a],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)]
    )

    # Stage A evaluation
    train_probs_a = model_a.predict(X_train_a)
    val_probs_a = model_a.predict(X_val_a)
    test_probs_a = model_a.predict(X_test_a)

    # Threshold optimization on validation set
    best_thresh_a = 0.5
    best_f1_a = 0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (val_probs_a >= t).astype(int)
        f1 = f1_score(y_val_a, preds)
        if f1 > best_f1_a:
            best_f1_a = f1
            best_thresh_a = t

    test_preds_a = (test_probs_a >= best_thresh_a).astype(int)
    a_prauc = average_precision_score(y_test_a, test_probs_a)
    a_prec = precision_score(y_test_a, test_preds_a)
    a_rec = recall_score(y_test_a, test_preds_a)
    a_f1 = f1_score(y_test_a, test_preds_a)

    print(f"\n  Stage A Results (Test):")
    print(f"    PR-AUC:    {a_prauc:.4f}")
    print(f"    Threshold: {best_thresh_a:.2f}")
    print(f"    Precision: {a_prec:.4f}")
    print(f"    Recall:    {a_rec:.4f}")
    print(f"    F1:        {a_f1:.4f}")

    model_a.save_model('model_stage_a_occurrence.txt')

    # ─────────────────────────────────────────────────────────
    # STAGE B: Conditional Intensity Model
    # ─────────────────────────────────────────────────────────
    print("\n[6/8] STAGE B: Training Conditional Intensity Model...")

    target_b = 'target_violation_count_1h'

    # Train ONLY on rows where target > 0
    train_active = train_df[train_df[target_b] > 0].copy()
    val_active = val_df[val_df[target_b] > 0].copy()

    X_train_b = train_active[features_full]
    y_train_b = train_active[target_b]
    X_val_b = val_active[features_full]
    y_val_b = val_active[target_b]
    X_test_b = test_df[features_full]  # predict on ALL test rows
    y_test_b = test_df[target_b]

    print(f"  Active training rows: {len(train_active):,} / {len(train_df):,}")
    print(f"  Active validation rows: {len(val_active):,} / {len(val_df):,}")

    # Optuna for Stage B (LightGBM Tweedie)
    print("  Running Optuna for LightGBM Tweedie (50 trials)...")

    def objective_b(trial):
        params = {
            'objective': 'tweedie',
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'metric': 'mae',
            'verbosity': -1,
            'n_jobs': -1,
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        
        dtrain = lgb.Dataset(X_train_b, y_train_b, categorical_feature=[c for c in cat_cols if c in features_full])
        dval = lgb.Dataset(X_val_b, y_val_b, reference=dtrain)
        
        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        
        val_preds = model.predict(X_val_b)
        return mean_absolute_error(y_val_b, val_preds)

    study_b = optuna.create_study(direction='minimize')
    study_b.optimize(objective_b, n_trials=50, show_progress_bar=False)

    best_params_b = study_b.best_params
    print(f"  Best MAE: {study_b.best_value:.4f}")

    # Train final Stage B LightGBM
    final_params_b = {
        'objective': 'tweedie',
        'metric': 'mae',
        'verbosity': -1,
        'n_jobs': -1,
        **best_params_b
    }

    dtrain_b = lgb.Dataset(X_train_b, y_train_b, categorical_feature=[c for c in cat_cols if c in features_full])
    dval_b = lgb.Dataset(X_val_b, y_val_b, reference=dtrain_b)

    model_b_lgb = lgb.train(
        final_params_b, dtrain_b,
        num_boost_round=1000,
        valid_sets=[dval_b],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)]
    )

    model_b_lgb.save_model('model_stage_b_lgb_intensity.txt')

    # CatBoost Stage B
    print("\n  Training CatBoost Tweedie regressor...")
    try:
        from catboost import CatBoostRegressor, Pool

        # Prepare categorical indices
        cat_indices = [features_full.index(c) for c in cat_cols if c in features_full]

        cb_model = CatBoostRegressor(
            loss_function='Tweedie:variance_power=1.5',
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            cat_features=cat_indices,
            verbose=100,
            early_stopping_rounds=50,
            random_seed=42
        )

        cb_model.fit(
            X_train_b, y_train_b,
            eval_set=(X_val_b, y_val_b),
            verbose=100
        )
        cb_model.save_model('model_stage_b_catboost_intensity.cbm')
        has_catboost = True
        print("  CatBoost training complete.")
    except ImportError:
        print("  CatBoost not installed. Using LightGBM only for Stage B.")
        has_catboost = False

    # ─────────────────────────────────────────────────────────
    # Ensemble: Weighted Blend
    # ─────────────────────────────────────────────────────────
    print("\n[7/8] Creating ensemble predictions...")

    # Stage B predictions
    lgb_test_preds = model_b_lgb.predict(X_test_b)
    lgb_test_preds = np.maximum(lgb_test_preds, 0)  # clip negative

    if has_catboost:
        cb_test_preds = cb_model.predict(X_test_b)
        cb_test_preds = np.maximum(cb_test_preds, 0)

        # Find optimal blend weight on validation set
        lgb_val_preds = model_b_lgb.predict(X_val_b)
        cb_val_preds = cb_model.predict(X_val_b)
        
        best_w = 0.5
        best_mae = float('inf')
        for w in np.arange(0.3, 0.8, 0.05):
            blend = w * lgb_val_preds + (1 - w) * cb_val_preds
            blend = np.maximum(blend, 0)
            mae = mean_absolute_error(y_val_b, blend)
            if mae < best_mae:
                best_mae = mae
                best_w = w

        print(f"  Optimal blend weight: LightGBM={best_w:.2f}, CatBoost={1-best_w:.2f}")
        stage_b_test_preds = best_w * lgb_test_preds + (1 - best_w) * cb_test_preds
    else:
        stage_b_test_preds = lgb_test_preds
        best_w = 1.0

    # ─────────────────────────────────────────────────────────
    # Combined Two-Stage Prediction
    # ─────────────────────────────────────────────────────────
    # expected_count = P(active) x E[count | active]
    combined_preds = test_probs_a * stage_b_test_preds

    # Risk Score (continuous ranking)
    test_df = test_df.copy()
    test_df['p_active'] = test_probs_a
    test_df['e_count_given_active'] = stage_b_test_preds
    test_df['expected_count'] = combined_preds

    # Normalize for risk score
    max_count = combined_preds.max() if combined_preds.max() > 0 else 1
    test_df['risk_score'] = (
        0.35 * test_probs_a +
        0.25 * np.clip(combined_preds / max_count, 0, 1) +
        0.20 * test_df['bayesian_critical_rate'].values +
        0.10 * np.clip(test_df['weighted_violation_count'].values / max(test_df['weighted_violation_count'].max(), 1), 0, 1) +
        0.10 * test_df['is_rush_hour'].values
    )

    # ─────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────
    print("\n[8/8] Evaluating combined system...")

    # Regression metrics
    combined_mae = mean_absolute_error(y_test_b, combined_preds)
    combined_rmse = np.sqrt(mean_squared_error(y_test_b, combined_preds))

    # Severity classification (for backwards comparison)
    def get_severity(x):
        if x <= 0.5: return 'CLEAR'
        elif x <= 2.5: return 'LOW'
        elif x <= 5.5: return 'MEDIUM'
        else: return 'CRITICAL'

    pred_severity = [get_severity(p) for p in combined_preds]
    true_severity = test_df['target_severity_1h'].values
    labels = ['CLEAR', 'LOW', 'MEDIUM', 'CRITICAL']
    macro_f1 = f1_score(true_severity, pred_severity, average='macro', labels=labels)

    # Critical class detection
    true_critical = (test_df[target_b] >= 6).astype(int).values
    pred_critical_prob = test_probs_a * (stage_b_test_preds >= 5.5).astype(float)
    
    # Recall@K (THE key operational metric)
    print("\n  Recall@K Analysis:")
    recall_at_k = {}
    for k in [10, 20, 50, 100]:
        # For each unique hour in test set, compute Recall@K
        recalls = []
        test_hours = test_df['hour'].unique()
        for h in test_hours:
            hour_mask = test_df['hour'] == h
            hour_true = test_df.loc[hour_mask, target_b].values
            hour_risk = test_df.loc[hour_mask, 'risk_score'].values

            if len(hour_true) < k:
                continue

            true_top_k_idx = np.argsort(-hour_true)[:k]
            pred_top_k_idx = np.argsort(-hour_risk)[:k]

            true_set = set(true_top_k_idx)
            pred_set = set(pred_top_k_idx)
            overlap = len(true_set & pred_set)
            recalls.append(overlap / k)

        recall_at_k[k] = np.mean(recalls) if recalls else 0
        print(f"    Recall@{k}: {recall_at_k[k]:.4f}")

    # Feature importance
    importance_a = pd.DataFrame({
        'Feature': model_a.feature_name(),
        'Importance_Gain': model_a.feature_importance(importance_type='gain')
    }).sort_values('Importance_Gain', ascending=False)

    importance_b = pd.DataFrame({
        'Feature': model_b_lgb.feature_name(),
        'Importance_Gain': model_b_lgb.feature_importance(importance_type='gain')
    }).sort_values('Importance_Gain', ascending=False)

    # Print summary
    print(f"\n  Combined System Results (Test Set):")
    print(f"    MAE:         {combined_mae:.4f}")
    print(f"    RMSE:        {combined_rmse:.4f}")
    print(f"    Macro F1:    {macro_f1:.4f}")
    print(f"    Stage A PR-AUC: {a_prauc:.4f}")

    # ─────────────────────────────────────────────────────────
    # Save comprehensive report
    # ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    report = f"""# Elite Model Evaluation Report

## Architecture
Two-Stage Zero-Inflated Model with Ensemble:
- **Stage A**: Hotspot Occurrence (Binary LightGBM, is_unbalance=True, Optuna-tuned)
- **Stage B**: Conditional Intensity (LightGBM Tweedie{' + CatBoost Tweedie blend' if has_catboost else ''})
- **Risk Score**: Composite ranking (no hard thresholds)

## Data Split (Chronological)
| Split | Rows | Time Range |
|-------|------|------------|
| Train (70%) | {len(train_df):,} | {train_df['hour'].min()} to {train_df['hour'].max()} |
| Validation (15%) | {len(val_df):,} | {val_df['hour'].min()} to {val_df['hour'].max()} |
| Test (15%) | {len(test_df):,} | {test_df['hour'].min()} to {test_df['hour'].max()} |

## Bayesian Features (Leakage-Free)
| Feature | Description |
|---------|-------------|
| `bayesian_active_rate` | Smoothed P(violation > 0) per grid |
| `bayesian_critical_rate` | Smoothed P(violation >= 6) per grid |
| `same_hour_dow_hist_avg` | Historical average for (grid, hour, DOW) |
| `is_grid_peak_hour` | Binary: is current hour the grid's historical peak? |
| `grid_avg_response_time` | Historical mean enforcement response time |

## Stage A: Hotspot Occurrence (Test Set)
| Metric | Value |
|--------|-------|
| PR-AUC | {a_prauc:.4f} |
| Optimal Threshold | {best_thresh_a:.2f} |
| Precision | {a_prec:.4f} |
| Recall | {a_rec:.4f} |
| F1 | {a_f1:.4f} |

## Stage B: Conditional Intensity
- **LightGBM Tweedie** trained on {len(train_active):,} active-only rows
- **Ensemble Blend Weight**: LightGBM={best_w:.2f}, CatBoost={1-best_w:.2f}

## Combined Two-Stage Results (Test Set)
| Metric | Value |
|--------|-------|
| MAE | {combined_mae:.4f} |
| RMSE | {combined_rmse:.4f} |
| Macro F1 (Severity) | {macro_f1:.4f} |

## Operational Metrics: Recall@K
*"If you dispatch K patrol teams to our top-K recommended grids, what fraction of true hotspots do you catch?"*

| K | Recall@K |
|---|----------|
| 10 | {recall_at_k.get(10, 0):.4f} |
| 20 | {recall_at_k.get(20, 0):.4f} |
| 50 | {recall_at_k.get(50, 0):.4f} |
| 100 | {recall_at_k.get(100, 0):.4f} |

## Top 10 Most Important Features

### Stage A (Occurrence)
| Rank | Feature | Importance (Gain) |
|------|---------|------------------|
"""
    for i, row in importance_a.head(10).iterrows():
        report += f"| {importance_a.index.get_loc(i)+1} | `{row['Feature']}` | {row['Importance_Gain']:.2f} |\n"

    report += """
### Stage B (Intensity)
| Rank | Feature | Importance (Gain) |
|------|---------|------------------|
"""
    for i, row in importance_b.head(10).iterrows():
        report += f"| {importance_b.index.get_loc(i)+1} | `{row['Feature']}` | {row['Importance_Gain']:.2f} |\n"

    report += f"""
## Optuna Hyperparameter Search
- Stage A: {study_a.best_value:.4f} PR-AUC (50 trials)
- Stage B: {study_b.best_value:.4f} MAE (50 trials)

## Models Saved
- `model_stage_a_occurrence.txt` (LightGBM binary)
- `model_stage_b_lgb_intensity.txt` (LightGBM Tweedie)
{'- `model_stage_b_catboost_intensity.cbm` (CatBoost Tweedie)' if has_catboost else ''}

## Feature List ({len(features_full)} total)
{chr(10).join(f'- `{f}`' for f in features_full)}

## Training Time
{elapsed:.1f} seconds
"""
    with open('elite_model_report.md', 'w') as f:
        f.write(report)

    # Save feature list and params for downstream use
    config = {
        'features': features_full,
        'cat_cols': cat_cols,
        'best_params_a': best_params_a,
        'best_params_b': best_params_b,
        'best_thresh_a': float(best_thresh_a),
        'blend_weight_lgb': float(best_w),
        'has_catboost': has_catboost,
        'bayesian_alpha': alpha_smooth,
        'global_active_rate': float(global_active_rate),
        'global_critical_rate': float(global_critical_rate),
    }
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print("PHASE 4 COMPLETE")
    print(f"{'='*70}")
    print(f"  Models saved: 2-3 model files")
    print(f"  Config saved: model_config.json")
    print(f"  Report saved: elite_model_report.md")
    print(f"  Time elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
