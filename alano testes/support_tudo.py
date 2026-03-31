import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import zipfile
import xml.etree.ElementTree as ET
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px

from collections import Counter
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

from catboost import (CatBoostRegressor,Pool,EFeaturesSelectionAlgorithm)
import shap
from itertools import product
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

'___________________________________________________________________________________'
DATE_COL = "Date"
TARGET_COL = "Revenue"
GROUP_COLS = ["Business_Unit", "Segment", "Subsegment"]

LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]

TOP_LEVEL = ["Business_Unit"]
MIDDLE_LEVEL = ["Business_Unit", "Segment"]
BOTTOM_LEVEL = ["Business_Unit", "Segment", "Subsegment"]


#correlation filter
def correlation_filter_train_test(X_train, X_test, threshold=0.8):
    # Keep only numeric columns
    X_train_num = X_train.select_dtypes(include="number")

    # Compute absolute Spearman correlation
    corr = X_train_num.corr(method="spearman").abs()

    # Keep only upper triangle
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Find columns to drop
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    # Drop the same columns from train and test
    X_train_filtered = X_train.drop(columns=to_drop, errors="ignore").copy()
    X_test_filtered = X_test.drop(columns=to_drop, errors="ignore").copy()

    return X_train_filtered, X_test_filtered, to_drop



# feature engineering functions
def create_base_features(X, group_cols):
    df = X.copy()

    df = df.sort_values(group_cols + [DATE_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["quarter"] = df[DATE_COL].dt.quarter

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["time_idx"] = df.groupby(group_cols).cumcount()
    df["series_id"] = df[group_cols].astype(str).agg("__".join, axis=1)

    cat_cols = group_cols + ["series_id"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df

def create_target_features(df, group_cols, target_col=TARGET_COL):
    data = df.copy()
    data = data.sort_values(group_cols + [DATE_COL]).reset_index(drop=True)

    for lag in LAGS:
        data[f"{target_col}_lag_{lag}"] = data.groupby(group_cols)[target_col].shift(lag)

    for window in ROLLING_WINDOWS:
        data[f"{target_col}_roll_mean_{window}"] = (
            data.groupby(group_cols)[target_col]
            .transform(lambda s: s.shift(1).rolling(window=window).mean())
        )

    for window in ROLLING_WINDOWS:
        data[f"{target_col}_roll_std_{window}"] = (
            data.groupby(group_cols)[target_col]
            .transform(lambda s: s.shift(1).rolling(window=window).std())
        )

    data[f"{target_col}_lag_diff_1"] = data[f"{target_col}_lag_1"] - data[f"{target_col}_lag_2"]
    data[f"{target_col}_lag_diff_12"] = data[f"{target_col}_lag_1"] - data[f"{target_col}_lag_12"]

    return data

def prepare_fold_for_feature_selection(X_train, y_train, X_test=None, y_test=None):
    X_train_base = create_base_features(X_train, GROUP_COLS)

    X_train_base[TARGET_COL] = pd.Series(y_train).reset_index(drop=True).values
    df_train = create_target_features(X_train_base, GROUP_COLS)
    df_train = create_advanced_revenue_features(df_train)

    df_train = df_train.dropna().reset_index(drop=True)

    y_train_final = df_train[TARGET_COL].copy()
    X_train_final = df_train.drop(columns=[TARGET_COL]).copy()

    train_dates = X_train_final[DATE_COL].copy()
    X_train_final = X_train_final.drop(columns=[DATE_COL])

    X_test_base = None
    if X_test is not None:
        X_test_base = create_base_features(X_test, GROUP_COLS)
        X_test_base = X_test_base.drop(columns=[DATE_COL])

    cat_features = ["Business_Unit", "Segment", "Subsegment", "series_id"]

    result = {
        "X_train": X_train_final,
        "y_train": y_train_final,
        "train_dates": train_dates,
        "X_test_base": X_test_base,
        "y_test": y_test,
        "cat_features": cat_features,
        "selected_feature_candidates": X_train_final.columns.tolist(),
    }

    return result

def prepare_all_folds_for_feature_selection(splits):
    prepared_folds = []

    for fold_id, (X_train, X_test, y_train, y_test) in enumerate(splits, start=1):
        fold_data = prepare_fold_for_feature_selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        fold_data["fold_id"] = fold_id
        prepared_folds.append(fold_data)

    return prepared_folds


# feature selection functions Mutual Information
def select_features_by_mi(prepared_folds, mi_threshold=0.0, top_k=None, random_state=42):
    results = []

    for fold in prepared_folds:
        fold_id = fold["fold_id"]
        X_train = fold["X_train"].copy()
        y_train = fold["y_train"].copy()
        cat_features = fold["cat_features"]

        # Numeric columns only
        numeric_cols = X_train.select_dtypes(include="number").columns.tolist()

        # Compute MI only for numeric features
        if len(numeric_cols) > 0:
            mi_scores = mutual_info_regression(X_train[numeric_cols], y_train, random_state=random_state)

            mi_df = pd.DataFrame({
                "feature": numeric_cols,
                "mi_score": mi_scores
            }).sort_values("mi_score", ascending=False).reset_index(drop=True)
        else:
            mi_df = pd.DataFrame(columns=["feature", "mi_score"])

        # Select numeric features
        if top_k is not None:
            numeric_keep = mi_df.head(top_k)["feature"].tolist()
        else:
            numeric_keep = mi_df.loc[mi_df["mi_score"] > mi_threshold, "feature"].tolist()

        # Always keep categorical features
        cat_keep = [col for col in cat_features if col in X_train.columns]

        selected_features = cat_keep + numeric_keep
        X_train_selected = X_train[selected_features].copy()

        results.append({
            "fold_id": fold_id,
            "X_train_selected": X_train_selected,
            "y_train": y_train,
            "cat_features": cat_keep,
            "selected_features": selected_features,
            "mi_ranking": mi_df
        })

    return results


# feature selection functions CatBoost
def catboost_select_features_fold(X_train,y_train,cat_features,total_features_to_keep=20):

    # Always keep hierarchy categorical features
    always_keep = [col for col in cat_features if col in X_train.columns]

    # Candidate features for recursive selection
    candidate_features = [col for col in X_train.columns if col not in always_keep]

    # If there is nothing to select, just return the categorical features
    if len(candidate_features) == 0:
        return always_keep, [], None

    # Number of extra features to keep besides always_keep
    extra_to_keep = total_features_to_keep - len(always_keep)
    extra_to_keep = max(0, min(extra_to_keep, len(candidate_features)))

    # If no extra features should be selected, keep only always_keep
    if extra_to_keep == 0:
        eliminated_features = candidate_features.copy()
        return always_keep, eliminated_features, None

    train_pool = Pool(data=X_train,label=y_train,cat_features=always_keep)

    model = CatBoostRegressor(loss_function="MAE",eval_metric="MAE",random_seed=42,verbose=0)

    selection = model.select_features(
        X=train_pool,
        features_for_select=candidate_features,
        num_features_to_select=extra_to_keep,
        steps=1,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
        train_final_model=False,
        logging_level="Silent"
    )

    selected_features = always_keep + selection["selected_features_names"]
    eliminated_features = selection["eliminated_features_names"]

    return selected_features, eliminated_features, selection


# permutation importance
def manual_permutation_importance_fold(X_train,y_train,cat_features,n_repeats=30,random_state=42):
    model = CatBoostRegressor(loss_function="MAE",eval_metric="MAE",random_seed=42,verbose=0,cat_features=cat_features)

    model.fit(X_train, y_train, verbose=0)

    baseline_pred = model.predict(X_train)
    baseline_mae = mean_absolute_error(y_train, baseline_pred)

    rng = np.random.default_rng(random_state)
    rows = []

    for col in X_train.columns:
        score_drops = []

        for _ in range(n_repeats):
            X_perm = X_train.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)

            perm_pred = model.predict(X_perm)
            perm_mae = mean_absolute_error(y_train, perm_pred)

            score_drops.append(perm_mae - baseline_mae)

        rows.append({
            "feature": col,
            "importance_mean": np.mean(score_drops),
            "importance_std": np.std(score_drops),
        })

    perm_df = pd.DataFrame(rows).sort_values("importance_mean",ascending=False).reset_index(drop=True)

    return perm_df, model, baseline_mae


# Final Features
def summarize_selected_features(results, method_name, feature_key="selected_features"):
    rows = []

    for result in results:
        fold_id = result["fold_id"]

        if feature_key in result:
            selected_features = result[feature_key]
        elif "X_train_selected" in result:
            selected_features = list(result["X_train_selected"].columns)
        else:
            selected_features = []

        for feature in selected_features:
            rows.append({
                "fold_id": fold_id,
                "feature": feature,
                "method": method_name,
                "selected": 1
            })

    return pd.DataFrame(rows)


# Model Preparation 

def build_test_row_features(row, history, group_cols):
    row_dict = row.to_dict()
    current_date = pd.to_datetime(row_dict[DATE_COL])

    row_dict["year"] = current_date.year
    row_dict["month"] = current_date.month
    row_dict["quarter"] = current_date.quarter
    row_dict["month_sin"] = np.sin(2 * np.pi * row_dict["month"] / 12)
    row_dict["month_cos"] = np.cos(2 * np.pi * row_dict["month"] / 12)

    row_dict["series_id"] = "__".join(str(row_dict[col]) for col in group_cols)

    mask = np.ones(len(history), dtype=bool)
    for col in group_cols:
        mask &= history[col].astype(str).values == str(row_dict[col])

    series_history = history.loc[mask].sort_values(DATE_COL).reset_index(drop=True)
    revenue_history = series_history[TARGET_COL].reset_index(drop=True)

    row_dict["time_idx"] = len(series_history)

    for lag in LAGS:
        row_dict[f"Revenue_lag_{lag}"] = (
            revenue_history.iloc[-lag] if len(revenue_history) >= lag else np.nan
        )

    for window in ROLLING_WINDOWS:
        row_dict[f"Revenue_roll_mean_{window}"] = (
            revenue_history.iloc[-window:].mean() if len(revenue_history) >= window else np.nan
        )

    for window in ROLLING_WINDOWS:
        row_dict[f"Revenue_roll_std_{window}"] = (
            revenue_history.iloc[-window:].std() if len(revenue_history) >= window else np.nan
        )

    lag_1 = row_dict.get("Revenue_lag_1", np.nan)
    lag_2 = row_dict.get("Revenue_lag_2", np.nan)
    lag_12 = row_dict.get("Revenue_lag_12", np.nan)

    row_dict["Revenue_lag_diff_1"] = lag_1 - lag_2 if pd.notna(lag_1) and pd.notna(lag_2) else np.nan
    row_dict["Revenue_lag_diff_12"] = lag_1 - lag_12 if pd.notna(lag_1) and pd.notna(lag_12) else np.nan

    features_df = pd.DataFrame([row_dict])
    features_df = create_advanced_revenue_features(features_df)
    return features_df


# Test hiperparameters

def generate_param_combinations(param_grid, max_combinations=None, random_state=42):
    """
    Gera combinações de hiperparâmetros a partir de um dicionário.
    Se max_combinations for definido, faz amostragem aleatória.
    """
    rng = np.random.default_rng(random_state)

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    all_combinations = [dict(zip(keys, combo)) for combo in product(*values)]

    if max_combinations is not None and max_combinations < len(all_combinations):
        idx = rng.choice(len(all_combinations), size=max_combinations, replace=False)
        all_combinations = [all_combinations[i] for i in idx]

    return all_combinations


def test_catboost_hyperparameters_middle_out(
    splits,
    selected_features,
    param_combinations,
    ranking_metric="bottom_mae"
):
    results = []

    for i, params in enumerate(param_combinations, start=1):
        fold_rows = []

        for fold_id, fold_split in enumerate(splits, start=1):
            model, test_middle_df, top_level_pred, bottom_eval, monthly_bottom, metrics = run_middle_out_fold(
                fold_split=fold_split,
                selected_features=selected_features,
                model_params=params
            )

            fold_rows.append({
                "fold_id": fold_id,
                **metrics
            })

        metrics_df = pd.DataFrame(fold_rows)

        row = {
            "trial": i,
            "params": params,
            "mean_bottom_mae": metrics_df["bottom_mae"].mean(),
            "mean_bottom_rmse": metrics_df["bottom_rmse"].mean(),
            "mean_monthly_mae": metrics_df["monthly_mae"].mean(),
            "mean_monthly_rmse": metrics_df["monthly_rmse"].mean(),
            "mean_abs_error_6m_total": metrics_df["abs_error_6m_total"].mean(),
        }

        results.append(row)

    results_df = pd.DataFrame(results)

    metric_map = {
        "bottom_mae": "mean_bottom_mae",
        "bottom_rmse": "mean_bottom_rmse",
        "monthly_mae": "mean_monthly_mae",
        "monthly_rmse": "mean_monthly_rmse",
        "abs_error_6m_total": "mean_abs_error_6m_total"
    }

    sort_col = metric_map[ranking_metric]
    results_df = results_df.sort_values(sort_col, ascending=True).reset_index(drop=True)

    return results_df





# Hierarchy Aggregation
def aggregate_fold_to_middle_level(X_part, y_part):
    df = X_part.copy()
    df[TARGET_COL] = pd.Series(y_part).values
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Numeric exogenous columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != TARGET_COL]

    # Aggregate rules
    agg_dict = {TARGET_COL: "sum"}
    for col in numeric_cols:
        agg_dict[col] = "first"

    middle_df = (
        df.groupby(MIDDLE_LEVEL + [DATE_COL], as_index=False)
        .agg(agg_dict)
        .sort_values(MIDDLE_LEVEL + [DATE_COL])
        .reset_index(drop=True)
    )

    X_middle = middle_df.drop(columns=[TARGET_COL])
    y_middle = middle_df[TARGET_COL]

    return X_middle, y_middle


def calculate_subsegment_shares(X_train_raw, y_train_raw):
    df = X_train_raw.copy()
    df[TARGET_COL] = pd.Series(y_train_raw).values
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    subsegment_revenue = (
        df.groupby(BOTTOM_LEVEL, as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "subsegment_revenue"})
    )

    segment_revenue = (
        df.groupby(MIDDLE_LEVEL, as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "segment_revenue"})
    )

    shares = subsegment_revenue.merge(
        segment_revenue,
        on=MIDDLE_LEVEL,
        how="left"
    )

    shares["share"] = shares["subsegment_revenue"] / shares["segment_revenue"]
    shares["share"] = shares["share"].fillna(0)

    return shares[BOTTOM_LEVEL + ["share"]]


def prepare_middle_train_fold(X_train_middle, y_train_middle):
    df_train = X_train_middle.copy()
    df_train[TARGET_COL] = pd.Series(y_train_middle).values

    df_train = create_base_features(df_train, MIDDLE_LEVEL)
    df_train = create_target_features(df_train, MIDDLE_LEVEL)
    df_train = create_advanced_revenue_features(df_train)
    df_train = df_train.dropna().reset_index(drop=True)

    y_train_final = df_train[TARGET_COL].copy()
    X_train_final = df_train.drop(columns=[TARGET_COL, DATE_COL]).copy()

    cat_features = MIDDLE_LEVEL + ["series_id"]

    return X_train_final, y_train_final, cat_features


def run_middle_out_fold(fold_split, selected_features, model_type="catboost", model_params=None):
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = fold_split

    X_train_middle, y_train_middle = aggregate_fold_to_middle_level(X_train_raw, y_train_raw)
    X_test_middle, y_test_middle = aggregate_fold_to_middle_level(X_test_raw, y_test_raw)

    X_train_full, y_train_final, cat_features = prepare_middle_train_fold(X_train_middle, y_train_middle)

    selected_features = [col for col in selected_features if col in X_train_full.columns]
    X_train_model = X_train_full[selected_features].copy()
    cat_features = [col for col in cat_features if col in selected_features]

    if model_params is None:
        model_params = {}

    if model_type == "catboost":
        default_params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": 0
        }
        default_params.update(model_params)
        model = CatBoostRegressor(**default_params)
        model.fit(X_train_model, y_train_final, cat_features=cat_features, verbose=0)

    elif model_type == "lightgbm":
        X_train_fit = X_train_model.copy()
        for col in X_train_fit.columns:
            if str(X_train_fit[col].dtype) in ["category", "object"]:
                X_train_fit[col] = X_train_fit[col].astype("category").cat.codes

        default_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "verbosity": -1
        }
        default_params.update(model_params)
        model = lgb.LGBMRegressor(**default_params)
        model.fit(X_train_fit, y_train_final)

    elif model_type == "random_forest":
        X_train_fit = X_train_model.copy()
        for col in X_train_fit.columns:
            if str(X_train_fit[col].dtype) in ["category", "object"]:
                X_train_fit[col] = X_train_fit[col].astype("category").cat.codes

        default_params = {
            "n_estimators": 200,
            "random_state": 42,
            "n_jobs": -1
        }
        default_params.update(model_params)
        model = RandomForestRegressor(**default_params)
        model.fit(X_train_fit, y_train_final)

    else:
        raise ValueError("model_type must be one of: 'catboost', 'lightgbm', 'random_forest'")

    history = X_train_middle.copy()
    history[TARGET_COL] = pd.Series(y_train_middle).values
    history[DATE_COL] = pd.to_datetime(history[DATE_COL])
    history = history.sort_values(MIDDLE_LEVEL + [DATE_COL]).reset_index(drop=True)

    test_middle_df = X_test_middle.copy()
    test_middle_df["y_true_middle"] = pd.Series(y_test_middle).values
    test_middle_df[DATE_COL] = pd.to_datetime(test_middle_df[DATE_COL])
    test_middle_df = test_middle_df.sort_values(MIDDLE_LEVEL + [DATE_COL]).reset_index(drop=True)

    preds = []

    for _, row in test_middle_df.iterrows():
        X_row = build_test_row_features(row, history, MIDDLE_LEVEL).reindex(columns=selected_features)

        if model_type in ["lightgbm", "random_forest"]:
            X_row_pred = X_row.copy()
            for col in X_row_pred.columns:
                if str(X_row_pred[col].dtype) in ["category", "object"]:
                    X_row_pred[col] = X_row_pred[col].astype("category").cat.codes
            y_pred = model.predict(X_row_pred)[0]
        else:
            y_pred = model.predict(X_row)[0]

        preds.append(y_pred)

        new_row = row.to_frame().T.copy()
        new_row[DATE_COL] = pd.to_datetime(new_row[DATE_COL])
        new_row[TARGET_COL] = y_pred
        history = pd.concat([history, new_row], ignore_index=True)

    test_middle_df["y_pred_middle"] = preds

    return {
        "model": model,
        "middle_predictions": test_middle_df,
        "X_train_raw": X_train_raw,
        "y_train_raw": y_train_raw,
        "X_test_raw": X_test_raw,
        "y_test_raw": y_test_raw
    }


## Rolling window

def test_training_windows_middle_out(
    df,
    window_sizes,
    selected_features,
    horizon=6,
    step=6,
    ranking_metric="monthly_mae",
    model_params=None
):
    results = []

    for train_window in window_sizes:
        splits = create_rolling_splits(
            df=df,
            target_col="Revenue",
            date_col="Date",
            train_window=train_window,
            horizon=horizon,
            step=step
        )

        fold_metrics = []

        for fold_id, fold_split in enumerate(splits, start=1):
            model, middle_pred, top_pred, bottom_eval, monthly_bottom, metrics = run_middle_out_fold(
                fold_split=fold_split,
                selected_features=selected_features,
                model_params=model_params
            )

            fold_metrics.append({
                "fold_id": fold_id,
                **metrics
            })

        metrics_df = pd.DataFrame(fold_metrics)

        results.append({
            "train_window": train_window,
            "n_folds": len(splits),
            "mean_bottom_mae": metrics_df["bottom_mae"].mean(),
            "mean_bottom_rmse": metrics_df["bottom_rmse"].mean(),
            "mean_monthly_mae": metrics_df["monthly_mae"].mean(),
            "mean_monthly_rmse": metrics_df["monthly_rmse"].mean(),
            "mean_abs_error_6m_total": metrics_df["abs_error_6m_total"].mean(),
            "std_monthly_mae": metrics_df["monthly_mae"].std(),
            "std_bottom_mae": metrics_df["bottom_mae"].std()
        })

    results_df = pd.DataFrame(results)

    metric_map = {
        "bottom_mae": "mean_bottom_mae",
        "bottom_rmse": "mean_bottom_rmse",
        "monthly_mae": "mean_monthly_mae",
        "monthly_rmse": "mean_monthly_rmse",
        "abs_error_6m_total": "mean_abs_error_6m_total"
    }

    results_df = results_df.sort_values(metric_map[ranking_metric], ascending=True).reset_index(drop=True)
    return results_df

def create_rolling_splits(df, target_col="Revenue", date_col="Date",
                          train_window=24, horizon=6, step=6):
    df = df.sort_values(date_col).reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    unique_dates = sorted(df[date_col].unique())
    splits = []

    for i in range(train_window, len(unique_dates) - horizon + 1, step):
        train_dates = unique_dates[i - train_window:i]
        test_dates = unique_dates[i:i + horizon]

        train_mask = X[date_col].isin(train_dates)
        test_mask = X[date_col].isin(test_dates)

        X_train = X[train_mask].copy()
        y_train = y[train_mask].copy()
        X_test = X[test_mask].copy()
        y_test = y[test_mask].copy()

        splits.append((X_train, X_test, y_train, y_test))

    return splits


## features extra

def create_advanced_revenue_features(df, eps=1e-9):
    data = df.copy()

    # Coefficient of variation
    data["Revenue_cv_3"] = data["Revenue_roll_std_3"] / (data["Revenue_roll_mean_3"] + eps)
    data["Revenue_cv_6"] = data["Revenue_roll_std_6"] / (data["Revenue_roll_mean_6"] + eps)
    data["Revenue_cv_12"] = data["Revenue_roll_std_12"] / (data["Revenue_roll_mean_12"] + eps)

    # Relative and deviation features
    data["Revenue_rel_to_roll12"] = data["Revenue_lag_1"] / (data["Revenue_roll_mean_12"] + eps)
    data["Revenue_dev_from_roll12"] = data["Revenue_lag_1"] - data["Revenue_roll_mean_12"]

    # Z-score relative to historical rolling window
    data["Revenue_zscore_12"] = (
        (data["Revenue_lag_1"] - data["Revenue_roll_mean_12"]) /
        (data["Revenue_roll_std_12"] + eps)
    )

    # Trend features
    data["trend_3_12"] = data["Revenue_roll_mean_3"] - data["Revenue_roll_mean_12"]
    data["trend_6_12"] = data["Revenue_roll_mean_6"] - data["Revenue_roll_mean_12"]

    # Year-over-year style features
    data["Revenue_yoy_ratio"] = data["Revenue_lag_1"] / (data["Revenue_lag_12"] + eps)
    data["Revenue_yoy_diff"] = data["Revenue_lag_1"] - data["Revenue_lag_12"]

    return data


# feature selection per fold

def build_consensus_features_by_fold(mi_results, catboost_results, permutation_results, always_keep=None):
    if always_keep is None:
        always_keep = ["Business_Unit", "Segment", "Subsegment", "series_id"]

    fold_feature_sets = []

    # assume same fold_ids exist in all result lists
    fold_ids = sorted([r["fold_id"] for r in mi_results])

    for fold_id in fold_ids:
        mi_fold = next(r for r in mi_results if r["fold_id"] == fold_id)
        cb_fold = next(r for r in catboost_results if r["fold_id"] == fold_id)
        perm_fold = next(r for r in permutation_results if r["fold_id"] == fold_id)

        mi_features = set(mi_fold.get("selected_features", []))
        cb_features = set(cb_fold.get("selected_features", []))
        perm_df = perm_fold["permutation_importance"].copy()

        # define permutation selected features:
        # here I use positive importance_mean as the rule
        perm_features = set(
            perm_df.loc[perm_df["importance_mean"] > 0, "feature"].tolist()
        )

        all_features = sorted(mi_features | cb_features | perm_features | set(always_keep))

        rows = []
        for feat in all_features:
            rows.append({
                "fold_id": fold_id,
                "feature": feat,
                "MI": 1.0 if feat in mi_features else 0.0,
                "CatBoost": 1.0 if feat in cb_features else 0.0,
                "Permutation": 1.0 if feat in perm_features else 0.0,
            })

        consensus_df = pd.DataFrame(rows)

        consensus_df["methods_supporting"] = (
            consensus_df["MI"].astype(int) +
            consensus_df["CatBoost"].astype(int) +
            consensus_df["Permutation"].astype(int)
        )

        consensus_df["keep_recommended"] = (
            consensus_df["feature"].isin(always_keep) |
            (consensus_df["CatBoost"] >= 1.0) |
            (consensus_df["methods_supporting"] >= 2)
        )

        consensus_df = consensus_df.sort_values(
            ["keep_recommended", "CatBoost", "Permutation", "MI", "feature"],
            ascending=[False, False, False, False, True]
        ).reset_index(drop=True)

        selected_features = consensus_df.loc[
            consensus_df["keep_recommended"], "feature"
        ].tolist()

        fold_feature_sets.append({
            "fold_id": fold_id,
            "consensus_df": consensus_df,
            "selected_features": selected_features
        })

    return fold_feature_sets


def build_model_folds_from_consensus(prepared_folds, fold_consensus):

    model_folds = []

    for prepared_fold in prepared_folds:
        fold_id = prepared_fold["fold_id"]
        consensus = next(c for c in fold_consensus if c["fold_id"] == fold_id)

        X_train_full = prepared_fold["X_train"].copy()
        y_train = prepared_fold["y_train"].copy()
        selected_features = consensus["selected_features"]

        selected_features = [f for f in selected_features if f in X_train_full.columns]
        cat_features = [c for c in prepared_fold["cat_features"] if c in selected_features]

        X_train_model = X_train_full[selected_features].copy()

        model_folds.append({
            "fold_id": fold_id,
            "X_train_model": X_train_model,
            "y_train": y_train,
            "selected_features": selected_features,
            "cat_features": cat_features,
            "consensus_df": consensus["consensus_df"]
        })

    return model_folds


## feature selection por fold

def build_catboost_results(prepared_folds, total_features_to_keep=20):
    catboost_results = []

    for fold in prepared_folds:
        fold_id = fold["fold_id"]
        X_train = fold["X_train"].copy()
        y_train = fold["y_train"].copy()
        cat_features = fold["cat_features"]

        selected_features, eliminated_features, selection = catboost_select_features_fold(
            X_train=X_train,
            y_train=y_train,
            cat_features=cat_features,
            total_features_to_keep=total_features_to_keep
        )

        catboost_results.append({
            "fold_id": fold_id,
            "selected_features": selected_features,
            "eliminated_features": eliminated_features,
            "selection": selection
        })

    return catboost_results




def build_permutation_results(prepared_folds, n_repeats=30, random_state=42):
    permutation_results = []

    for fold in prepared_folds:
        fold_id = fold["fold_id"]
        X_train = fold["X_train"].copy()
        y_train = fold["y_train"].copy()
        cat_features = fold["cat_features"]

        perm_df, model, baseline_mae = manual_permutation_importance_fold(
            X_train=X_train,
            y_train=y_train,
            cat_features=cat_features,
            n_repeats=n_repeats,
            random_state=random_state
        )

        permutation_results.append({
            "fold_id": fold_id,
            "permutation_importance": perm_df,
            "model": model,
            "baseline_mae": baseline_mae
        })

    return permutation_results



## Feature selection LGM

def select_features_by_lightgbm(prepared_folds,importance_threshold=0.0,top_k=None,n_estimators=200,learning_rate=0.05,
    num_leaves=31,random_state=42,add_segment_code=False,segment_col="Segment",segment_code_col="Segment_code",plot_top_n=None,
    figsize=(10, 8)):

    results = []

    for fold in prepared_folds:
        fold_id = fold["fold_id"]
        X_train = fold["X_train"].copy()
        y_train = fold["y_train"].copy()
        cat_features = fold.get("cat_features", [])

        # opcional: criar Segment_code
        if add_segment_code:
            if segment_col not in X_train.columns:
                raise KeyError(
                    f"Fold {fold_id}: column '{segment_col}' not found in X_train."
                )

            X_train[segment_code_col] = (
                X_train[segment_col].astype("category").cat.codes
            )

        # LightGBM só aceita números diretamente neste setup
        # por isso convertemos categóricas para codes
        X_model = X_train.copy()

        for col in X_model.columns:
            if str(X_model[col].dtype) == "category" or X_model[col].dtype == "object":
                X_model[col] = X_model[col].astype("category").cat.codes

        model_fs = lgb.LGBMRegressor(n_estimators=n_estimators,learning_rate=learning_rate,num_leaves=num_leaves,random_state=random_state,
            verbosity=-1)

        model_fs.fit(X_model, y_train)

        importance_df = pd.DataFrame({
            "feature": X_model.columns,
            "importance": model_fs.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        if top_k is not None:
            selected_features = importance_df.head(top_k)["feature"].tolist()
        else:
            selected_features = (
                importance_df.loc[
                    importance_df["importance"] > importance_threshold, "feature"
                ].tolist()
            )

        # manter apenas features que existem no X_train original
        selected_features = [f for f in selected_features if f in X_train.columns]

        # opcionalmente garantir que cat_features ficam sempre
        always_keep = [col for col in cat_features if col in X_train.columns]
        selected_features = list(dict.fromkeys(always_keep + selected_features))

        X_train_selected = X_train[selected_features].copy()

        if plot_top_n is not None and plot_top_n > 0:
            plt.figure(figsize=figsize)
            importance_df.head(plot_top_n).plot(
                kind="barh",
                x="feature",
                y="importance",
                legend=False
            )
            plt.title(f"Fold {fold_id} - Top {plot_top_n} Feature Importances")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        results.append({
            "fold_id": fold_id,
            "X_train_selected": X_train_selected,
            "y_train": y_train,
            "cat_features": [col for col in cat_features if col in selected_features],
            "selected_features": selected_features,
            "lgbm_ranking": importance_df,
            "model": model_fs
        })

    return results


def add_segment_code_from_train(train_df, test_df=None, segment_col="Segment", code_col="Segment_code"):
    train_out = train_df.copy()

    categories = pd.Index(train_out[segment_col].dropna().unique())
    mapping = {cat: i for i, cat in enumerate(categories)}

    train_out[code_col] = train_out[segment_col].map(mapping).fillna(-1).astype(int)

    if test_df is None:
        return train_out, mapping

    test_out = test_df.copy()
    test_out[code_col] = test_out[segment_col].map(mapping).fillna(-1).astype(int)

    return train_out, test_out, mapping


## random forest feature importance

def select_features_by_random_forest(prepared_folds,importance_threshold=0.0,top_k=None,n_estimators=200,random_state=42,
    n_jobs=-1,add_segment_code=False,segment_col="Segment",segment_code_col="Segment_code",plot_top_n=None,figsize=(10, 8)):

    results = []

    for fold in prepared_folds:
        fold_id = fold["fold_id"]
        X_train = fold["X_train"].copy()
        y_train = fold["y_train"].copy()
        cat_features = fold.get("cat_features", [])

        if add_segment_code:
            if segment_col not in X_train.columns:
                raise KeyError(
                    f"Fold {fold_id}: column '{segment_col}' not found in X_train."
                )
            X_train[segment_code_col] = (
                X_train[segment_col].astype("category").cat.codes
            )

        X_model = X_train.copy()

        for col in X_model.columns:
            if str(X_model[col].dtype) == "category" or X_model[col].dtype == "object":
                X_model[col] = X_model[col].astype("category").cat.codes

        model_rf = RandomForestRegressor(n_estimators=n_estimators,random_state=random_state,n_jobs=n_jobs)

        model_rf.fit(X_model, y_train)

        importance_df = pd.DataFrame({
            "feature": X_model.columns,
            "importance": model_rf.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        if top_k is not None:
            selected_features = importance_df.head(top_k)["feature"].tolist()
        else:
            selected_features = (
                importance_df.loc[
                    importance_df["importance"] > importance_threshold, "feature"
                ].tolist()
            )

        selected_features = [f for f in selected_features if f in X_train.columns]

        always_keep = [col for col in cat_features if col in X_train.columns]
        selected_features = list(dict.fromkeys(always_keep + selected_features))

        X_train_selected = X_train[selected_features].copy()

        if plot_top_n is not None and plot_top_n > 0:
            plt.figure(figsize=figsize)
            importance_df.head(plot_top_n).sort_values("importance").plot(
                kind="barh",
                x="feature",
                y="importance",
                legend=False
            )
            plt.title(f"Fold {fold_id} - Top {plot_top_n} Feature Importances (RF)")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.show()

        results.append({
            "fold_id": fold_id,
            "X_train_selected": X_train_selected,
            "y_train": y_train,
            "cat_features": [col for col in cat_features if col in selected_features],
            "selected_features": selected_features,
            "rf_ranking": importance_df,
            "model": model_rf
        })

    return results


## Models

### ETS


def fit_ets_forecasts_by_segment(seg_train,clean_segments,segment_col="Segment",date_col="Period",target_col="Revenue",
    forecast_horizon=6,min_obs_full_seasonality=24,min_obs_half_seasonality=12,full_seasonal_periods=12,half_seasonal_periods=6,
    trend="add",seasonal="add",optimized=True):
 
    ets_forecasts = {}
    ets_models = {}
    summary_rows = []

    for seg in clean_segments:
        seg_df = (
            seg_train.loc[seg_train[segment_col] == seg]
            .sort_values(date_col)
            .copy()
        )

        train_data = seg_df[target_col].values

        if len(train_data) == 0:
            summary_rows.append({
                "segment": seg,
                "n_obs": 0,
                "trend": None,
                "seasonal": None,
                "seasonal_periods": None,
                "status": "skipped_empty"
            })
            continue

        try:
            if len(train_data) >= min_obs_full_seasonality:
                model = ExponentialSmoothing(
                    train_data,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=full_seasonal_periods
                )
                seasonal_used = seasonal
                seasonal_periods_used = full_seasonal_periods

            elif len(train_data) >= min_obs_half_seasonality:
                model = ExponentialSmoothing(
                    train_data,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=half_seasonal_periods
                )
                seasonal_used = seasonal
                seasonal_periods_used = half_seasonal_periods

            else:
                model = ExponentialSmoothing(
                    train_data,
                    trend=trend,
                    seasonal=None
                )
                seasonal_used = None
                seasonal_periods_used = None

            fitted = model.fit(optimized=optimized)
            forecast = fitted.forecast(forecast_horizon)

            ets_models[seg] = fitted
            ets_forecasts[seg] = forecast

            summary_rows.append({
                "segment": seg,
                "n_obs": len(train_data),
                "trend": trend,
                "seasonal": seasonal_used,
                "seasonal_periods": seasonal_periods_used,
                "status": "ok"
            })

        except Exception as e:
            summary_rows.append({
                "segment": seg,
                "n_obs": len(train_data),
                "trend": trend,
                "seasonal": seasonal if len(train_data) >= min_obs_half_seasonality else None,
                "seasonal_periods": (
                    full_seasonal_periods if len(train_data) >= min_obs_full_seasonality
                    else half_seasonal_periods if len(train_data) >= min_obs_half_seasonality
                    else None
                ),
                "status": f"error: {str(e)}"
            })

    summary_df = pd.DataFrame(summary_rows)

    print(f"ETS fitted for {len(ets_forecasts)} segments")

    return {
        "forecasts": ets_forecasts,
        "models": ets_models,
        "summary": summary_df
    }


def calculate_forecast_metrics(y_true, y_pred, prefix):
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    denom = np.abs(y_true).sum()
    wape = (np.abs(y_true - y_pred).sum() / denom * 100) if denom != 0 else np.nan

    return {
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_wape": wape
    }

def evaluate_middle_out_predictions(middle_pred_df, X_train_raw, y_train_raw, X_test_raw, y_test_raw):

    middle_pred_df = middle_pred_df.copy()
    middle_pred_df[DATE_COL] = pd.to_datetime(middle_pred_df[DATE_COL])

    # -------- TOP LEVEL --------
    top_pred = (
        middle_pred_df.groupby(TOP_LEVEL + [DATE_COL], as_index=False)[
            ["y_true_middle", "y_pred_middle"]
        ]
        .sum()
        .rename(columns={
            "y_true_middle": "y_true_top",
            "y_pred_middle": "y_pred_top"
        })
    )

    # -------- BOTTOM LEVEL --------
    shares = calculate_subsegment_shares(X_train_raw, y_train_raw)

    bottom_pred = middle_pred_df[MIDDLE_LEVEL + [DATE_COL, "y_pred_middle"]].merge(
        shares,
        on=MIDDLE_LEVEL,
        how="left"
    )

    bottom_pred["share"] = bottom_pred["share"].fillna(0)
    bottom_pred["y_pred_bottom"] = bottom_pred["y_pred_middle"] * bottom_pred["share"]

    actual_bottom = X_test_raw.copy()
    actual_bottom[TARGET_COL] = pd.Series(y_test_raw).values
    actual_bottom[DATE_COL] = pd.to_datetime(actual_bottom[DATE_COL])

    actual_bottom = (
        actual_bottom.groupby(BOTTOM_LEVEL + [DATE_COL], as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "y_true_bottom"})
    )

    bottom_eval = actual_bottom.merge(
        bottom_pred[BOTTOM_LEVEL + [DATE_COL, "y_pred_bottom"]],
        on=BOTTOM_LEVEL + [DATE_COL],
        how="left"
    )

    bottom_eval["y_pred_bottom"] = bottom_eval["y_pred_bottom"].fillna(0)

    # -------- METRICS --------
    metrics = {}
    metrics.update(
        calculate_forecast_metrics(
            middle_pred_df["y_true_middle"],
            middle_pred_df["y_pred_middle"],
            "middle"
        )
    )
    metrics.update(
        calculate_forecast_metrics(
            top_pred["y_true_top"],
            top_pred["y_pred_top"],
            "top"
        )
    )
    metrics.update(
        calculate_forecast_metrics(
            bottom_eval["y_true_bottom"],
            bottom_eval["y_pred_bottom"],
            "bottom"
        )
    )

    return {
        "middle_predictions": middle_pred_df,
        "top_predictions": top_pred,
        "bottom_predictions": bottom_eval,
        "metrics": metrics
    }


def run_and_evaluate_middle_out_fold(fold_split, selected_features, model_type="catboost", model_params=None):
    model_result = run_middle_out_fold(
        fold_split=fold_split,
        selected_features=selected_features,
        model_type=model_type,
        model_params=model_params
    )

    eval_result = evaluate_middle_out_predictions(
        middle_pred_df=model_result["middle_predictions"],
        X_train_raw=model_result["X_train_raw"],
        y_train_raw=model_result["y_train_raw"],
        X_test_raw=model_result["X_test_raw"],
        y_test_raw=model_result["y_test_raw"]
    )

    return {
        "model": model_result["model"],
        "middle_predictions": eval_result["middle_predictions"],
        "top_predictions": eval_result["top_predictions"],
        "bottom_predictions": eval_result["bottom_predictions"],
        "metrics": eval_result["metrics"]
    }



def evaluate_middle_out_folds(splits, selected_features, model_type="catboost", model_params=None):
    fold_metrics = []
    fold_outputs = []

    for fold_id, fold_split in enumerate(splits, start=1):
        result = run_and_evaluate_middle_out_fold(
            fold_split=fold_split,
            selected_features=selected_features,
            model_type=model_type,
            model_params=model_params
        )

        fold_metrics.append({
            "fold_id": fold_id,
            **result["metrics"]
        })

        fold_outputs.append({
            "fold_id": fold_id,
            "model": result["model"],
            "middle_predictions": result["middle_predictions"],
            "top_predictions": result["top_predictions"],
            "bottom_predictions": result["bottom_predictions"]
        })

    metrics_df = pd.DataFrame(fold_metrics)
    return metrics_df, fold_outputs


def run_prophet_middle_out_fold(fold_split, yearly_seasonality=True):
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = fold_split

    X_train_middle, y_train_middle = aggregate_fold_to_middle_level(X_train_raw, y_train_raw)
    X_test_middle, y_test_middle = aggregate_fold_to_middle_level(X_test_raw, y_test_raw)

    train_df = X_train_middle.copy()
    train_df[TARGET_COL] = pd.Series(y_train_middle).values
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    test_df = X_test_middle.copy()
    test_df["y_true_middle"] = pd.Series(y_test_middle).values
    test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL])

    train_df["unique_id"] = (
        train_df["Business_Unit"].astype(str) + "__" +
        train_df["Segment"].astype(str)
    )
    test_df["unique_id"] = (
        test_df["Business_Unit"].astype(str) + "__" +
        test_df["Segment"].astype(str)
    )

    prophet_forecasts = []

    n_future = test_df[DATE_COL].nunique()

    for unique_id in train_df["unique_id"].unique():
        train_series = train_df[train_df["unique_id"] == unique_id].copy()
        test_series = test_df[test_df["unique_id"] == unique_id].copy()

        if train_series.empty or test_series.empty:
            continue

        prophet_train = train_series.rename(columns={
            DATE_COL: "ds",
            TARGET_COL: "y"
        })[["ds", "y"]]

        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(prophet_train)

        future = model.make_future_dataframe(
            periods=n_future,
            freq="MS",
            include_history=False
        )
        forecast = model.predict(future)

        forecast["unique_id"] = unique_id
        prophet_forecasts.append(
            forecast[["unique_id", "ds", "yhat"]]
        )

    fcst_df = pd.concat(prophet_forecasts, ignore_index=True)

    middle_predictions = test_df.rename(columns={DATE_COL: "ds"}).merge(
        fcst_df,
        on=["unique_id", "ds"],
        how="left"
    )

    middle_predictions = middle_predictions.rename(columns={
        "ds": DATE_COL,
        "yhat": "y_pred_middle"
    })[
        ["Business_Unit", "Segment", DATE_COL, "y_true_middle", "y_pred_middle"]
    ].copy()

    return {
        "middle_predictions": middle_predictions,
        "X_train_raw": X_train_raw,
        "y_train_raw": y_train_raw,
        "X_test_raw": X_test_raw,
        "y_test_raw": y_test_raw
    }




def run_ets_middle_out_fold(
    fold_split,
    trend="add",
    seasonal="add",
    min_obs_full_seasonality=24,
    min_obs_half_seasonality=12,
    full_seasonal_periods=12,
    half_seasonal_periods=6,
    optimized=True
):
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = fold_split

    X_train_middle, y_train_middle = aggregate_fold_to_middle_level(X_train_raw, y_train_raw)
    X_test_middle, y_test_middle = aggregate_fold_to_middle_level(X_test_raw, y_test_raw)

    train_df = X_train_middle.copy()
    train_df[TARGET_COL] = pd.Series(y_train_middle).values
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])

    test_df = X_test_middle.copy()
    test_df["y_true_middle"] = pd.Series(y_test_middle).values
    test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL])

    train_df["unique_id"] = (
        train_df["Business_Unit"].astype(str) + "__" +
        train_df["Segment"].astype(str)
    )
    test_df["unique_id"] = (
        test_df["Business_Unit"].astype(str) + "__" +
        test_df["Segment"].astype(str)
    )

    ets_forecasts = []

    for unique_id in train_df["unique_id"].unique():
        train_series = (
            train_df[train_df["unique_id"] == unique_id]
            .sort_values(DATE_COL)
            .copy()
        )
        test_series = (
            test_df[test_df["unique_id"] == unique_id]
            .sort_values(DATE_COL)
            .copy()
        )

        if train_series.empty or test_series.empty:
            continue

        y_train_series = train_series[TARGET_COL].values
        horizon = len(test_series)

        if len(y_train_series) >= min_obs_full_seasonality:
            model = ExponentialSmoothing(
                y_train_series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=full_seasonal_periods
            )
        elif len(y_train_series) >= min_obs_half_seasonality:
            model = ExponentialSmoothing(
                y_train_series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=half_seasonal_periods
            )
        else:
            model = ExponentialSmoothing(
                y_train_series,
                trend=trend,
                seasonal=None
            )

        fitted = model.fit(optimized=optimized)
        forecast = fitted.forecast(horizon)

        seg_fcst = test_series[["Business_Unit", "Segment", DATE_COL]].copy()
        seg_fcst["y_pred_middle"] = forecast
        ets_forecasts.append(seg_fcst)

    middle_predictions = (
        test_df[["Business_Unit", "Segment", DATE_COL, "y_true_middle"]]
        .merge(
            pd.concat(ets_forecasts, ignore_index=True),
            on=["Business_Unit", "Segment", DATE_COL],
            how="left"
        )
        .copy()
    )

    return {
        "middle_predictions": middle_predictions,
        "X_train_raw": X_train_raw,
        "y_train_raw": y_train_raw,
        "X_test_raw": X_test_raw,
        "y_test_raw": y_test_raw
    }   


def evaluate_statistical_middle_out_folds(splits, model_runner, **model_kwargs):
    fold_metrics = []
    fold_outputs = []

    for fold_id, fold_split in enumerate(splits, start=1):
        model_result = model_runner(fold_split=fold_split, **model_kwargs)

        eval_result = evaluate_middle_out_predictions(
            middle_pred_df=model_result["middle_predictions"],
            X_train_raw=model_result["X_train_raw"],
            y_train_raw=model_result["y_train_raw"],
            X_test_raw=model_result["X_test_raw"],
            y_test_raw=model_result["y_test_raw"]
        )

        fold_metrics.append({
            "fold_id": fold_id,
            **eval_result["metrics"]
        })

        fold_outputs.append({
            "fold_id": fold_id,
            "middle_predictions": eval_result["middle_predictions"],
            "top_predictions": eval_result["top_predictions"],
            "bottom_predictions": eval_result["bottom_predictions"]
        })

    metrics_df = pd.DataFrame(fold_metrics)
    return metrics_df, fold_outputs


























