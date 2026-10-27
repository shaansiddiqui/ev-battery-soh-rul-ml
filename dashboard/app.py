from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="EV Battery Health Dashboard",
    page_icon="🔋",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_LABELS = {
    "mean_voltage": "Mean Voltage (V)",
    "voltage_headroom": "Voltage Headroom (V)",
    "mean_current": "Mean Current (A)",
    "mean_temperature": "Mean Temperature (°C)",
    "max_temperature": "Maximum Temperature (°C)",
    "discharge_duration": "Discharge Duration (s)"
}


# ------------------------------------------------------------
# Load data and model
# ------------------------------------------------------------
@st.cache_resource
def load_assets():
    feature_df = pd.read_csv(
        RESULTS_DIR / "all_batteries_soh_features.csv"
    )

    prediction_df = pd.read_csv(
        RESULTS_DIR / "soh_predictions_all_models.csv"
    )

    comparison_df = pd.read_csv(
        RESULTS_DIR / "soh_model_comparison.csv"
    )

    importance_df = pd.read_csv(
        RESULTS_DIR / "shap_feature_importance.csv"
    )

    with open(
        MODELS_DIR / "best_soh_model_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    model = joblib.load(
        MODELS_DIR / "best_soh_model.joblib"
    )

    return (
        feature_df,
        prediction_df,
        comparison_df,
        importance_df,
        metadata,
        model
    )


try:
    (
        feature_df,
        prediction_df,
        comparison_df,
        importance_df,
        metadata,
        model
    ) = load_assets()

except FileNotFoundError:
    st.error(
        "Required model/result files are missing. "
        "Run model_comparison_soh.py and explain_soh_xgboost.py first."
    )
    st.stop()


feature_columns = metadata["feature_columns"]
selected_model_name = metadata["selected_model"]


# ------------------------------------------------------------
# Dashboard heading
# ------------------------------------------------------------
st.title("🔋 EV Battery Health Monitoring Dashboard")

st.caption(
    "Explainable machine-learning dashboard using the NASA PCoE "
    "Li-ion Battery Aging Dataset."
)

st.info(
    f"Selected SOH model: **{selected_model_name}** | "
    f"Validation method: **Leave-One-Battery-Out (LOBO)**"
)


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("Battery and Cycle Selection")

battery_ids = sorted(feature_df["battery_id"].unique())

selected_battery = st.sidebar.selectbox(
    "Select Battery",
    battery_ids
)

battery_df = (
    feature_df[
        feature_df["battery_id"] == selected_battery
    ]
    .sort_values("cycle_number")
    .reset_index(drop=True)
)

cycle_numbers = battery_df["cycle_number"].tolist()

selected_cycle = st.sidebar.select_slider(
    "Select Discharge Cycle",
    options=cycle_numbers,
    value=cycle_numbers[-1]
)

selected_row = battery_df[
    battery_df["cycle_number"] == selected_cycle
].iloc[0]


# ------------------------------------------------------------
# Selected-cycle SOH prediction
# ------------------------------------------------------------
selected_prediction_df = prediction_df[
    (prediction_df["model"] == selected_model_name)
    & (prediction_df["test_battery"] == selected_battery)
    & (prediction_df["cycle_number"] == selected_cycle)
]

actual_soh = selected_row["SOH"]

if not selected_prediction_df.empty:
    predicted_soh = selected_prediction_df[
        "predicted_soh"
    ].iloc[0]

    absolute_error = selected_prediction_df[
        "absolute_error"
    ].iloc[0]
else:
    # Fallback prediction from final model trained on all data.
    prediction_input = pd.DataFrame(
        [selected_row[feature_columns].to_dict()]
    )

    predicted_soh = model.predict(prediction_input)[0]
    absolute_error = abs(actual_soh - predicted_soh)


st.subheader(
    f"Battery {selected_battery} — Cycle {selected_cycle}"
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "Actual SOH",
    f"{actual_soh:.2f}%"
)

metric_col2.metric(
    "Predicted SOH",
    f"{predicted_soh:.2f}%"
)

metric_col3.metric(
    "Absolute Error",
    f"{absolute_error:.2f} percentage points"
)

metric_col4.metric(
    "End-of-Life Threshold",
    "80.00%"
)

if actual_soh <= 80:
    st.warning(
        "This cycle is at or below the defined 80% SOH "
        "end-of-life threshold."
    )
else:
    st.success(
        "This cycle remains above the defined 80% SOH "
        "end-of-life threshold."
    )


# ------------------------------------------------------------
# SOH trend for selected battery
# ------------------------------------------------------------
st.subheader("SOH Trend: Actual vs LOBO Prediction")

battery_predictions = prediction_df[
    (prediction_df["model"] == selected_model_name)
    & (prediction_df["test_battery"] == selected_battery)
].sort_values("cycle_number")

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    battery_predictions["cycle_number"],
    battery_predictions["actual_soh"],
    color="black",
    linewidth=2.5,
    label="Actual SOH"
)

ax.plot(
    battery_predictions["cycle_number"],
    battery_predictions["predicted_soh"],
    color="#dc2626",
    linewidth=2,
    label=f"{selected_model_name} Prediction"
)

ax.axhline(
    80,
    color="#f59e0b",
    linestyle="--",
    linewidth=1.5,
    label="80% EOL Threshold"
)

ax.scatter(
    selected_cycle,
    actual_soh,
    color="#2563eb",
    s=80,
    zorder=5,
    label="Selected Cycle"
)

ax.set_xlabel("Discharge Cycle Number")
ax.set_ylabel("SOH (%)")
ax.set_title(
    f"{selected_battery}: Actual vs Predicted SOH"
)
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Prediction trend uses leave-one-battery-out results: "
    "the selected battery was not included in training for this plot."
)


# ------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------
st.subheader("Model Comparison: Mean LOBO Validation Performance")

comparison_display = comparison_df.copy()

comparison_display = comparison_display.rename(
    columns={
        "model": "Model",
        "RMSE_percent": "RMSE (%)",
        "MAE_percent": "MAE (%)",
        "R2": "R²"
    }
)

comparison_display = comparison_display.round(3)

st.dataframe(
    comparison_display,
    use_container_width=True,
    hide_index=True
)

best_result = comparison_df.iloc[0]

st.success(
    f"{best_result['model']} was selected because it achieved the "
    f"lowest mean LOBO MAE: {best_result['MAE_percent']:.3f}% "
    f"and mean R²: {best_result['R2']:.3f}."
)


# ------------------------------------------------------------
# Feature values for selected cycle
# ------------------------------------------------------------
st.subheader("Sensor Features for Selected Cycle")

feature_display = pd.DataFrame({
    "Feature": [
        FEATURE_LABELS[feature]
        for feature in feature_columns
    ],
    "Value": [
        selected_row[feature]
        for feature in feature_columns
    ]
})

feature_display["Value"] = feature_display["Value"].round(4)

st.dataframe(
    feature_display,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# Global SHAP explanation
# ------------------------------------------------------------
st.subheader("Global Explainable AI: Feature Importance")

importance_display = importance_df.copy()

importance_display["feature"] = importance_display[
    "feature"
].map(FEATURE_LABELS)

importance_display = importance_display.rename(
    columns={
        "feature": "Feature",
        "mean_absolute_shap_value": "Mean |SHAP Value|"
    }
)

importance_display = importance_display.sort_values(
    "Mean |SHAP Value|",
    ascending=True
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    importance_display["Feature"],
    importance_display["Mean |SHAP Value|"],
    color="#e11d48"
)

ax.set_xlabel("Mean Absolute SHAP Value")
ax.set_title(
    "Global Feature Importance for SOH Prediction"
)
ax.grid(axis="x", alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Higher mean absolute SHAP value means the feature has a larger "
    "average influence on SOH predictions. SHAP shows model associations, "
    "not direct physical causation."
)


# ------------------------------------------------------------
# Local SHAP explanation
# ------------------------------------------------------------
st.subheader("Local Explainable AI: Why This SOH Was Predicted")

selected_features_df = pd.DataFrame(
    [selected_row[feature_columns].to_dict()]
)

explainer = shap.TreeExplainer(model)

local_shap_values = explainer(selected_features_df)

local_explanation = pd.DataFrame({
    "Feature": [
        FEATURE_LABELS[feature]
        for feature in feature_columns
    ],
    "Feature Value": selected_features_df.iloc[0].values,
    "SHAP Contribution": local_shap_values.values[0]
})

local_explanation["Absolute Contribution"] = (
    local_explanation["SHAP Contribution"].abs()
)

local_explanation = local_explanation.sort_values(
    "Absolute Contribution",
    ascending=True
)

bar_colors = [
    "#e11d48" if value > 0 else "#2563eb"
    for value in local_explanation["SHAP Contribution"]
]

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    local_explanation["Feature"],
    local_explanation["SHAP Contribution"],
    color=bar_colors
)

ax.axvline(0, color="black", linewidth=1)

ax.set_xlabel("SHAP Contribution to Predicted SOH (%)")
ax.set_title(
    f"Feature Contributions: {selected_battery}, Cycle {selected_cycle}"
)
ax.grid(axis="x", alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Red bars push the SOH prediction upward relative to the model baseline; "
    "blue bars push it downward."
)

local_display = local_explanation.sort_values(
    "Absolute Contribution",
    ascending=False
)[
    ["Feature", "Feature Value", "SHAP Contribution"]
].round(4)

st.dataframe(
    local_display,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# Limitations
# ------------------------------------------------------------
with st.expander("Model Scope and Limitations"):
    st.markdown(
        """
- The dashboard uses NASA PCoE experimental battery-aging data.
- LOBO validation tests generalization to an unseen battery among the four available cells.
- The dataset is small, so results should not be treated as real-world EV deployment performance.
- SHAP explains model behavior and feature association; it does not prove physical causation.
- Real-hardware validation is planned for the major-project phase.
        """
    )