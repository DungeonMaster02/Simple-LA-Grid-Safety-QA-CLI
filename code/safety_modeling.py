import xgboost
from safety_model_prepare import prepare_safety_data
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import pandas as pd

data = prepare_safety_data(["property_crime", "violence_crime"]) #containing pannel, property crime and violence crime data
df_property = data["property_crime"].copy()
df_violence = data["violence_crime"].copy()
results = list()

folds = [
    {
        "name": "fold1",
        "train_start": "2010-01-01",
        "train_end":   "2019-12-31",
        "val_start":   "2020-01-01",
        "val_end":     "2020-12-31",
    },
    {
        "name": "fold2",
        "train_start": "2010-01-01",
        "train_end":   "2020-12-31",
        "val_start":   "2021-01-01",
        "val_end":     "2021-12-31",
    },
    {
        "name": "fold3",
        "train_start": "2010-01-01",
        "train_end":   "2021-12-31",
        "val_start":   "2022-01-01",
        "val_end":     "2022-12-31",
    },
    {
        "name": "fold4",
        "train_start": "2010-01-01",
        "train_end":   "2022-12-31",
        "val_start":   "2023-01-01",
        "val_end":     "2023-12-31",
    },
    {
        "name": "fold5",
        "train_start": "2010-01-01",
        "train_end":   "2023-12-31",
        "val_start":   "2024-01-01",
        "val_end":     "2024-11-30", # 2024 dec is dropped since the model uses current month data to predict next month
    },
]

# property crime model preparation
property_target = "property_crime_target_next"
px = [c for c in df_property.columns if c not in ["grid_id", "month_start", "property_crime_target_next"]]
# Violence crime model preparation
violence_target = "violence_crime_target_next"
vx = [c for c in df_violence.columns if c not in ["grid_id", "month_start", "violence_crime_target_next"]]

# Training
for fold in folds:
    tstart = fold["train_start"]
    tend = fold["train_end"]
    vstart = fold["val_start"]
    vend = fold["val_end"]
    p_train = df_property[(df_property["month_start"] >= tstart) & (df_property["month_start"] <= tend)]
    p_val = df_property[(df_property["month_start"] >= vstart) & (df_property["month_start"] <= vend)]
    v_train = df_violence[(df_violence["month_start"] >= tstart) & (df_violence["month_start"] <= tend)]
    v_val = df_violence[(df_violence["month_start"] >= vstart) & (df_violence["month_start"] <= vend)]

    px_train = p_train[px]
    py_train = p_train[property_target]
    px_val = p_val[px]
    py_val = p_val[property_target]
    vx_train = v_train[vx]
    vy_train = v_train[violence_target]
    vx_val = v_val[vx]
    vy_val = v_val[violence_target]

    pdtrain_reg = xgboost.DMatrix(px_train, py_train, enable_categorical=True) # convert data into DMatrix format for better performance
    pdval_reg = xgboost.DMatrix(px_val, py_val, enable_categorical=True)
    vdtrain_reg = xgboost.DMatrix(vx_train, vy_train, enable_categorical=True)
    vdval_reg = xgboost.DMatrix(vx_val, vy_val, enable_categorical=True)

    pparams = {"objective": "reg:squarederror", "tree_method": "hist", "eval_metric": "rmse", "max_depth": 4, "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "seed": 42, "gamma": 1, "min_child_weight": 10, "reg_alpha": 0.5,  "reg_lambda": 2.0}
    vparams = {"objective": "reg:squarederror", "tree_method": "hist", "eval_metric": "rmse", "max_depth": 4, "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "seed": 42, "gamma": 1, "min_child_weight": 10}
    n = 1000
    p_evals = [(pdtrain_reg, "train"), (pdval_reg, "validation")]
    v_evals = [(vdtrain_reg, "train"), (vdval_reg, "validation")]

    p_model = xgboost.train(
        params=pparams,
        dtrain=pdtrain_reg,
        num_boost_round=n,
        evals = p_evals,
        verbose_eval=50,
        early_stopping_rounds=50
    )
    v_model = xgboost.train(
        params=vparams,
        dtrain=vdtrain_reg,
        num_boost_round=n,
        evals = v_evals,
        verbose_eval=50,
        early_stopping_rounds=50
    )
    
    #model evaluation
    p_preds = p_model.predict(pdval_reg)
    v_preds = v_model.predict(vdval_reg)
    p_rmse = root_mean_squared_error(py_val, p_preds) #rmse = sqrt(mean_squared_error), set squared=False to get RMSE instead of MSE
    v_rmse = root_mean_squared_error(vy_val, v_preds) 
    p_mae = mean_absolute_error(py_val, p_preds) #mae = mean absolute error between true values and predictions
    v_mae = mean_absolute_error(vy_val, v_preds)
    print(f"{fold['name']} - property Crime RMSE: {p_rmse:.4f}, MAE: {p_mae:.4f} | Violence Crime RMSE: {v_rmse:.4f}, MAE: {v_mae:.4f}")

    #get result
    p_result = p_val[["grid_id", "month_start", property_target]].copy()
    p_result["pred_property"] = p_preds

    v_result = v_val[["grid_id", "month_start", violence_target]].copy()
    v_result["pred_violence"] = v_preds

    combined = p_result.merge(
        v_result,
        on=["grid_id", "month_start"],
        how="inner"
    )

    combined["actual_weighted_score"] = (
        0.4 * combined[property_target] +
        0.6 * combined[violence_target]
    )

    combined["pred_weighted_score"] = (
        0.4 * combined["pred_property"] +
        0.6 * combined["pred_violence"]
    )

    combined_rmse = root_mean_squared_error(
        combined["actual_weighted_score"],
        combined["pred_weighted_score"]
    )
    combined_mae = mean_absolute_error(
        combined["actual_weighted_score"],
        combined["pred_weighted_score"]
    )

    annual = combined.groupby("grid_id", as_index=False).agg(
    actual_property_annual=(property_target, "sum"),
    pred_property_annual=("pred_property", "sum"),
    actual_violence_annual=(violence_target, "sum"),
    pred_violence_annual=("pred_violence", "sum"),
    )

    annual["actual_risk"] = (
        0.4 * annual["actual_property_annual"] +
        0.6 * annual["actual_violence_annual"]
    )

    annual["pred_risk"] = (
        0.4 * annual["pred_property_annual"] +
        0.6 * annual["pred_violence_annual"]
    )

    k = int(np.ceil(len(annual) * 0.1)) # top 10% high risk cells
    actual_hotspots = set(
        annual.nlargest(k, "actual_risk")["grid_id"]
    )

    pred_hotspots = set(
        annual.nlargest(k, "pred_risk")["grid_id"]
    )
    hit_rate = len(actual_hotspots & pred_hotspots) / len(actual_hotspots)
    jaccard = len(actual_hotspots & pred_hotspots) / len(actual_hotspots | pred_hotspots)
    print(
        f"{fold['name']} | "
        f"property RMSE: {p_rmse:.4f}, MAE: {p_mae:.4f} | "
        f"Violence RMSE: {v_rmse:.4f}, MAE: {v_mae:.4f} | "
        f"Combined RMSE: {combined_rmse:.4f}, MAE: {combined_mae:.4f} | "
        f"Hit Rate: {hit_rate:.4f}, Jaccard: {jaccard:.4f}"
    )

    results.append({
        "fold": fold["name"],
        "property_rmse": p_rmse,
        "property_mae": p_mae,
        "violence_rmse": v_rmse,
        "violence_mae": v_mae,
        "combined_rmse": combined_rmse,
        "combined_mae": combined_mae,
        "hit_rate": hit_rate,
        "jaccard": jaccard,
        "n_rows": len(combined),
    })

results_df = pd.DataFrame(results)
results_df.to_csv("../output/safety_folds_results.csv", index=False)
print("\nCross-validation summary:")
print(results_df)

print("\nAverage metrics:")
print(results_df[[
    "property_rmse", "property_mae",
    "violence_rmse", "violence_mae",
    "combined_rmse", "combined_mae",
    "hit_rate", "jaccard"   
]].mean())
p_val["month"] = p_val["month_start"].dt.to_period("M")

print(p_val.groupby("month").size())