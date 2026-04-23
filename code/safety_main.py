import os
import numpy as np
import pandas as pd
import xgboost
from safety_model_prepare import prepare_safety_data


def train_safety_models(train_end: str = "2024-11-30"):
    """Train property-crime and violence-crime models."""
    data = prepare_safety_data(["property_crime", "violence_crime"])
    panel = data["panel"].copy()
    df_property = data["property_crime"].copy()
    df_violence = data["violence_crime"].copy()

    train_end_ts = pd.Timestamp(train_end)
    p_train = df_property[df_property["month_start"] <= train_end_ts].copy()
    v_train = df_violence[df_violence["month_start"] <= train_end_ts].copy()

    if p_train.empty or v_train.empty:
        raise ValueError("No training rows found. Please check monthly panel data range.")

    property_target = "property_crime_target_next"
    violence_target = "violence_crime_target_next"

    p_features = [
        c for c in p_train.columns
        if c not in ["grid_id", "month_start", property_target]
    ]
    v_features = [
        c for c in v_train.columns
        if c not in ["grid_id", "month_start", violence_target]
    ]

    pdtrain_reg = xgboost.DMatrix(
        p_train[p_features],
        p_train[property_target],
        enable_categorical=True
    )
    vdtrain_reg = xgboost.DMatrix(
        v_train[v_features],
        v_train[violence_target],
        enable_categorical=True
    )

    pparams = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
        "max_depth": 4,
        "eta": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "seed": 42,
        "gamma": 1,
        "min_child_weight": 10,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    }
    vparams = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
        "max_depth": 4,
        "eta": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "seed": 42,
        "gamma": 1,
        "min_child_weight": 10,
    }

    p_model = xgboost.train(params=pparams, dtrain=pdtrain_reg, num_boost_round=600)
    v_model = xgboost.train(params=vparams, dtrain=vdtrain_reg, num_boost_round=600)

    return panel, p_model, v_model, p_features, v_features


def ensure_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
    return out[feature_cols].fillna(0.0)


def build_feature_for_month(panel: pd.DataFrame, current_month: pd.Timestamp, target_col: str) -> pd.DataFrame:
    """
    Build one-step-ahead feature rows for all grids at current_month.
    The model will use these rows to predict current_month + 1.
    """
    current_month = pd.Timestamp(current_month).replace(day=1)
    feat = panel[panel["month_start"] == current_month].copy()

    if feat.empty:
        raise ValueError(f"No rows found for month_start={current_month.date()}.")

    lag_base = panel[["grid_id", "month_start", target_col]].copy()
    tmp_cols = []
    for lag in range(1, 13):
        shifted = lag_base.copy()
        shifted["month_start"] = shifted["month_start"] + pd.DateOffset(months=lag)
        col_name = f"{target_col}_lag{lag}_tmp"
        shifted = shifted.rename(columns={target_col: col_name})
        feat = feat.merge(shifted, on=["grid_id", "month_start"], how="left")
        tmp_cols.append(col_name)

    feat[f"{target_col}_lag1"] = feat[f"{target_col}_lag1_tmp"]
    feat[f"{target_col}_lag2"] = feat[f"{target_col}_lag2_tmp"]
    feat[f"{target_col}_lag3"] = feat[f"{target_col}_lag3_tmp"]
    feat[f"{target_col}_lag6"] = feat[f"{target_col}_lag6_tmp"]
    feat[f"{target_col}_lag12"] = feat[f"{target_col}_lag12_tmp"]

    feat[f"{target_col}_roll3"] = feat[
        [f"{target_col}_lag{i}_tmp" for i in range(1, 4)]
    ].mean(axis=1)
    feat[f"{target_col}_roll6"] = feat[
        [f"{target_col}_lag{i}_tmp" for i in range(1, 7)]
    ].mean(axis=1)
    feat[f"{target_col}_roll12"] = feat[
        [f"{target_col}_lag{i}_tmp" for i in range(1, 13)]
    ].mean(axis=1)

    feat.drop(columns=tmp_cols, inplace=True)
    return feat


def predict_future_monthly(
    panel: pd.DataFrame,
    p_model,
    v_model,
    p_features: list[str],
    v_features: list[str],
    pred_start: str = "2025-01-01",
    pred_end: str = "2026-12-01",
) -> pd.DataFrame:
    """Recursive monthly prediction from pred_start to pred_end."""
    work_panel = panel.copy()
    work_panel["month_start"] = pd.to_datetime(work_panel["month_start"]).dt.to_period("M").dt.to_timestamp()
    work_panel = work_panel.sort_values(["grid_id", "month_start"]).reset_index(drop=True)

    dynamic_cols = {
        "month_start",
        "crime_count",
        "property_crime",
        "violence_crime",
        "month",
        "year",
        "month_sin",
        "month_cos",
        "time_idx",
    }
    static_cols = [c for c in work_panel.columns if c not in dynamic_cols and c != "grid_id"]
    static_grid = (
        work_panel.sort_values("month_start")
        .drop_duplicates(subset=["grid_id"], keep="first")[["grid_id"] + static_cols]
        .copy()
    )
    base_year = int(work_panel["month_start"].dt.year.min())

    pred_months = pd.date_range(pred_start, pred_end, freq="MS")
    monthly_results = []

    for pred_month in pred_months:
        current_month = pred_month - pd.DateOffset(months=1)

        p_feat = build_feature_for_month(work_panel, current_month, "property_crime")
        v_feat = build_feature_for_month(work_panel, current_month, "violence_crime")

        px = ensure_feature_columns(p_feat, p_features)
        vx = ensure_feature_columns(v_feat, v_features)

        pdtest_reg = xgboost.DMatrix(px, enable_categorical=True)
        vdtest_reg = xgboost.DMatrix(vx, enable_categorical=True)

        pred_property = np.clip(p_model.predict(pdtest_reg), a_min=0, a_max=None)
        pred_violence = np.clip(v_model.predict(vdtest_reg), a_min=0, a_max=None)

        pred_month_df = pd.DataFrame(
            {
                "grid_id": p_feat["grid_id"].astype(int).to_numpy(),
                "month_start": pd.Timestamp(pred_month),
                "pred_property": pred_property,
                "pred_violence": pred_violence,
            }
        )
        monthly_results.append(pred_month_df)

        next_rows = static_grid.copy()
        next_rows["month_start"] = pd.Timestamp(pred_month)
        next_rows["property_crime"] = pred_property
        next_rows["violence_crime"] = pred_violence
        next_rows["crime_count"] = pred_property + pred_violence

        month_num = int(pred_month.month)
        year_num = int(pred_month.year)
        next_rows["month"] = month_num
        next_rows["year"] = year_num
        next_rows["month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
        next_rows["month_cos"] = np.cos(2 * np.pi * month_num / 12.0)
        next_rows["time_idx"] = (year_num - base_year) * 12 + month_num

        next_rows = next_rows[work_panel.columns]
        work_panel = pd.concat([work_panel, next_rows], ignore_index=True)

    return pd.concat(monthly_results, ignore_index=True)


def build_grid_safety_2026(future_monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 2026 predictions to annual grid safety score."""
    out = future_monthly.copy()
    out["month_start"] = pd.to_datetime(out["month_start"])
    pred_2026 = out[out["month_start"].dt.year == 2026].copy()

    annual = pred_2026.groupby("grid_id", as_index=False).agg(
        pred_property_annual=("pred_property", "sum"),
        pred_violence_annual=("pred_violence", "sum"),
    )

    annual["risk"] = (
        0.4 * annual["pred_property_annual"] +
        0.6 * annual["pred_violence_annual"]
    )

    annual["risk_pct"] = annual["risk"].rank(pct=True, method="average")
    annual["safety_score"] = (100 * (1 - annual["risk_pct"])).round(2)
    annual["risk_norm"] = annual["risk_pct"]

    annual["risk_rank"] = annual["risk"].rank(method="first", ascending=False).astype(int)
    topk = int(np.ceil(len(annual) * 0.1))
    annual["hotspot_2026_top10pct"] = (annual["risk_rank"] <= topk).astype(int)

    annual["safety_level"] = pd.cut(
        annual["safety_score"],
        bins=[-np.inf, 20, 40, 60, 80, np.inf],
        labels=["Very Unsafe", "Unsafe", "Moderate", "Safe", "Very Safe"],
    )

    annual = annual.sort_values("risk_rank").reset_index(drop=True)
    return annual[
        [
            "grid_id",
            "pred_property_annual",
            "pred_violence_annual",
            "risk",
            "risk_norm",
            "safety_score",
            "safety_level",
            "risk_rank",
            "hotspot_2026_top10pct",
        ]
    ]


def save_grid_safety_to_csv(final_df: pd.DataFrame, output_dir: str = "../output"):
    """Save grid safety result for downstream usage without database dependency."""
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(f"{output_dir}/grid_scores.csv", index=False)


if __name__ == "__main__":
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)

    panel, p_model, v_model, p_features, v_features = train_safety_models(train_end="2024-11-30")

    future_monthly = predict_future_monthly(
        panel=panel,
        p_model=p_model,
        v_model=v_model,
        p_features=p_features,
        v_features=v_features,
        pred_start="2025-01-01",
        pred_end="2026-12-01",
    )
    future_monthly.to_csv(f"{output_dir}/future_monthly_predictions_2025_to_2026.csv", index=False)

    final_grid_safety = build_grid_safety_2026(future_monthly)
    final_grid_safety.to_csv(f"{output_dir}/final_grid_safety_2026.csv", index=False)

    save_grid_safety_to_csv(final_grid_safety, output_dir=output_dir)
    print("2026 grid safety prediction finished and saved to CSV files.")
