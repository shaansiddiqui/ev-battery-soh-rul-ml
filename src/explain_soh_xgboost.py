"""
Explain the selected SOH model using SHAP.

This script loads the best SOH model saved by model_comparison_soh.py
and creates global and local XAI explanations.

Run:
    python src/explain_soh_xgboost.py
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_DATA_FILE = RESULTS_DIR / "all_batteries_soh_features.csv"
MODEL_FILE = MODELS_DIR / "best_soh_model.joblib"
METADATA_FILE = MODELS_DIR / "best_soh_model_metadata.json"

# A single cycle for local explanation.
# You can change this later from the dashboard.
SELECTED_BATTERY = "B0018"
SELECTED_CYCLE = 75


def save_current_plot(file_name):
    """Save current Matplotlib plot and close it."""
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / file_name,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------
    # 1. Check required files
    # ------------------------------------------------------------
    if not FEATURE_DATA_FILE.exists():
        raise FileNotFoundError(
            "Feature dataset not found. Run model_comparison_soh.py first."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Saved SOH model not found. Run model_comparison_soh.py first."
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Model metadata not found. Run model_comparison_soh.py first."
        )

    # ------------------------------------------------------------
    # 2. Load trained best model and features
    # ------------------------------------------------------------
    df = pd.read_csv(FEATURE_DATA_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]
    selected_model_name = metadata["selected_model"]

    model = joblib.load(MODEL_FILE)

    X = df[feature_columns].copy()

    print(f"Loaded selected model: {selected_model_name}")
    print(f"Number of cycles explained: {len(X)}")
    print("Features used:")
    print(feature_columns)

    # ------------------------------------------------------------
    # 3. Generate SHAP values
    # ------------------------------------------------------------
    # TreeExplainer is optimized for XGBoost and other tree models.
    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X)

    # ------------------------------------------------------------
    # 4. Global feature importance
    # ------------------------------------------------------------
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "mean_absolute_shap_value": mean_abs_shap
    }).sort_values(
        "mean_absolute_shap_value",
        ascending=False
    )

    importance_df.to_csv(
        RESULTS_DIR / "shap_feature_importance.csv",
        index=False
    )

    print("\nGLOBAL SHAP FEATURE IMPORTANCE")
    print(importance_df.to_string(index=False))

    # SHAP bar chart
    shap.plots.bar(
        shap_values,
        max_display=len(feature_columns),
        show=False
    )

    plt.title(
        f"Global Feature Importance for SOH Prediction ({selected_model_name})"
    )

    save_current_plot("shap_global_feature_importance.png")

    # ------------------------------------------------------------
    # 5. SHAP summary / beeswarm plot
    # ------------------------------------------------------------
    shap.plots.beeswarm(
        shap_values,
        max_display=len(feature_columns),
        show=False
    )

    plt.title(
        f"SHAP Summary Plot for SOH Prediction ({selected_model_name})"
    )

    save_current_plot("shap_summary_beeswarm.png")

    # ------------------------------------------------------------
    # 6. Local explanation for one selected battery cycle
    # ------------------------------------------------------------
    local_df = df[
        (df["battery_id"] == SELECTED_BATTERY)
        & (df["cycle_number"] == SELECTED_CYCLE)
    ].copy()

    if local_df.empty:
        raise ValueError(
            f"No row found for {SELECTED_BATTERY}, cycle {SELECTED_CYCLE}."
        )

    selected_index = local_df.index[0]

    # SHAP values are based on X row position, so find its position.
    shap_position = X.index.get_loc(selected_index)

    local_shap_values = shap_values[shap_position]
    local_features = X.loc[[selected_index]]

    predicted_soh = model.predict(local_features)[0]
    actual_soh = local_df["SOH"].iloc[0]

    local_explanation_df = pd.DataFrame({
        "feature": feature_columns,
        "feature_value": local_features.iloc[0].values,
        "shap_contribution_to_soh": local_shap_values.values
    })

    local_explanation_df["absolute_shap_contribution"] = (
        local_explanation_df["shap_contribution_to_soh"].abs()
    )

    local_explanation_df = local_explanation_df.sort_values(
        "absolute_shap_contribution",
        ascending=False
    )

    local_explanation_df.to_csv(
        RESULTS_DIR / (
            f"shap_local_explanation_"
            f"{SELECTED_BATTERY}_cycle_{SELECTED_CYCLE}.csv"
        ),
        index=False
    )

    print("\nLOCAL EXPLANATION")
    print(f"Battery: {SELECTED_BATTERY}")
    print(f"Cycle: {SELECTED_CYCLE}")
    print(f"Actual SOH: {actual_soh:.2f}%")
    print(f"Predicted SOH: {predicted_soh:.2f}%")
    print(local_explanation_df.to_string(index=False))

    # Waterfall chart: explains one individual prediction.
    shap.plots.waterfall(
        local_shap_values,
        max_display=len(feature_columns),
        show=False
    )

    plt.title(
        f"Why the Model Predicted {predicted_soh:.2f}% SOH\n"
        f"{SELECTED_BATTERY}, Cycle {SELECTED_CYCLE}"
    )

    save_current_plot(
        f"shap_local_waterfall_"
        f"{SELECTED_BATTERY}_cycle_{SELECTED_CYCLE}.png"
    )

    # ------------------------------------------------------------
    # 7. Save compact prediction summary for dashboard usage
    # ------------------------------------------------------------
    local_summary = pd.DataFrame([{
        "battery_id": SELECTED_BATTERY,
        "cycle_number": SELECTED_CYCLE,
        "actual_soh_percent": actual_soh,
        "predicted_soh_percent": predicted_soh,
        "absolute_error_percent": abs(actual_soh - predicted_soh),
        "selected_model": selected_model_name
    }])

    local_summary.to_csv(
        RESULTS_DIR / "shap_selected_cycle_prediction.csv",
        index=False
    )

    print("\nSaved XAI outputs:")
    print("results/shap_feature_importance.csv")
    print("results/shap_global_feature_importance.png")
    print("results/shap_summary_beeswarm.png")
    print(
        f"results/shap_local_explanation_"
        f"{SELECTED_BATTERY}_cycle_{SELECTED_CYCLE}.csv"
    )
    print(
        f"results/shap_local_waterfall_"
        f"{SELECTED_BATTERY}_cycle_{SELECTED_CYCLE}.png"
    )
    print("results/shap_selected_cycle_prediction.csv")


if __name__ == "__main__":
    main()