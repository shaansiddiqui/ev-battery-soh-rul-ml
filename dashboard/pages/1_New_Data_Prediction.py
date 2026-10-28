from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


st.set_page_config(
    page_title="New Battery Data Prediction",
    page_icon="📤",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


@st.cache_resource
def load_assets():
    feature_df = pd.read_csv(
        RESULTS_DIR / "all_batteries_soh_features.csv"
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

    return feature_df, metadata, model


def get_health_status(predicted_soh):
    """Return dashboard health category and message."""
    if predicted_soh >= 90:
        return "Healthy", "✅", (
            "SOH is above 90%. Continue normal monitoring."
        )

    if predicted_soh >= 80:
        return "Monitor", "⚠️", (
            "SOH is between 80% and 90%. Monitor battery ageing closely."
        )

    return "End of Life", "🔴", (
        "SOH is at or below 80%. Maintenance or replacement should be considered."
    )


def prepare_uploaded_data(raw_df, feature_columns, training_df):
    """
    Validate uploaded/manual input and check whether values are within
    the observed NASA training-data range.
    """
    data = raw_df.copy()

    # Allow users to upload min_voltage and cutoff_voltage instead.
    if (
        "voltage_headroom" not in data.columns
        and "min_voltage" in data.columns
        and "cutoff_voltage" in data.columns
    ):
        data["voltage_headroom"] = (
            pd.to_numeric(data["min_voltage"], errors="coerce")
            - pd.to_numeric(data["cutoff_voltage"], errors="coerce")
        )

    missing_columns = [
        feature for feature in feature_columns
        if feature not in data.columns
    ]

    if missing_columns:
        return None, (
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    input_df = data[feature_columns].copy()

    for feature in feature_columns:
        input_df[feature] = pd.to_numeric(
            input_df[feature],
            errors="coerce"
        )

    if input_df.isna().any().any():
        invalid_columns = input_df.columns[
            input_df.isna().any()
        ].tolist()

        return None, (
            "These columns contain missing or non-numeric values: "
            + ", ".join(invalid_columns)
        )

    # Use observed 1st–99th percentile range for practical input checks.
    compatibility_notes = []

    for _, row in input_df.iterrows():
        out_of_range = []

        for feature in feature_columns:
            lower_bound = training_df[feature].quantile(0.01)
            upper_bound = training_df[feature].quantile(0.99)

            value = row[feature]

            if value < lower_bound or value > upper_bound:
                out_of_range.append(
                    FEATURE_LABELS[feature]
                )

        if out_of_range:
            compatibility_notes.append(
                "Review input: outside training range for "
                + ", ".join(out_of_range)
            )
        else:
            compatibility_notes.append(
                "Compatible with observed training-data range"
            )

    input_df["input_compatibility"] = compatibility_notes

    return input_df, None


def show_local_shap(model, feature_row):
    """Display local SHAP explanation for one new prediction."""
    explainer = shap.TreeExplainer(model)

    local_shap_values = explainer(feature_row)

    explanation_df = pd.DataFrame({
        "Feature": [
            FEATURE_LABELS[column]
            for column in feature_row.columns
        ],
        "Input Value": feature_row.iloc[0].values,
        "SHAP Contribution": local_shap_values.values[0]
    })

    explanation_df["Absolute Contribution"] = (
        explanation_df["SHAP Contribution"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "Absolute Contribution",
        ascending=True
    )

    colours = [
        "#e11d48" if value > 0 else "#2563eb"
        for value in explanation_df["SHAP Contribution"]
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        explanation_df["Feature"],
        explanation_df["SHAP Contribution"],
        color=colours
    )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP Contribution to Predicted SOH (%)")
    ax.set_title("Why the Model Produced This SOH Prediction")
    ax.grid(axis="x", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    explanation_df = explanation_df.sort_values(
        "Absolute Contribution",
        ascending=False
    )

    st.dataframe(
        explanation_df[
            ["Feature", "Input Value", "SHAP Contribution"]
        ].round(4),
        use_container_width=True,
        hide_index=True
    )


try:
    training_df, metadata, model = load_assets()

except FileNotFoundError:
    st.error(
        "Required model files are missing. Run "
        "`model_comparison_soh.py` and `explain_soh_xgboost.py` first."
    )
    st.stop()


feature_columns = metadata["feature_columns"]

st.title("📤 New Battery Data Prediction")

st.caption(
    "Upload cycle-level battery features or manually enter sensor values "
    "to receive an SOH prediction, health status, input-quality check, "
    "and SHAP explanation."
)

st.info(
    "Important: This model was trained on NASA discharge data where "
    "discharge current is negative. Real-hardware data must use the "
    "same units and sign convention before prediction."
)

st.subheader("Required Input Features")

required_features_df = pd.DataFrame({
    "CSV Column": feature_columns,
    "Meaning": [
        FEATURE_LABELS[column]
        for column in feature_columns
    ]
})

st.dataframe(
    required_features_df,
    use_container_width=True,
    hide_index=True
)

# Downloadable template
template_df = pd.DataFrame([
    training_df[feature_columns].median().to_dict()
])

st.download_button(
    label="Download CSV Input Template",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="battery_cycle_input_template.csv",
    mime="text/csv"
)

input_mode = st.radio(
    "Choose Input Method",
    ["Upload CSV", "Manual Entry"],
    horizontal=True
)

raw_input_df = None

if input_mode == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload cycle-level battery feature CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        raw_input_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(
            raw_input_df.head(),
            use_container_width=True
        )

else:
    st.subheader("Manual Feature Entry")

    median_values = training_df[feature_columns].median()

    with st.form("manual_prediction_form"):
        column_1, column_2 = st.columns(2)

        with column_1:
            mean_voltage = st.number_input(
                "Mean Voltage (V)",
                value=float(median_values["mean_voltage"]),
                format="%.4f"
            )

            voltage_headroom = st.number_input(
                "Voltage Headroom (V)",
                value=float(median_values["voltage_headroom"]),
                format="%.4f"
            )

            mean_current = st.number_input(
                "Mean Current (A)",
                value=float(median_values["mean_current"]),
                format="%.4f"
            )

        with column_2:
            mean_temperature = st.number_input(
                "Mean Temperature (°C)",
                value=float(median_values["mean_temperature"]),
                format="%.4f"
            )

            max_temperature = st.number_input(
                "Maximum Temperature (°C)",
                value=float(median_values["max_temperature"]),
                format="%.4f"
            )

            discharge_duration = st.number_input(
                "Discharge Duration (s)",
                value=float(median_values["discharge_duration"]),
                format="%.2f"
            )

        submitted = st.form_submit_button("Predict SOH")

    if submitted:
        raw_input_df = pd.DataFrame([{
            "mean_voltage": mean_voltage,
            "voltage_headroom": voltage_headroom,
            "mean_current": mean_current,
            "mean_temperature": mean_temperature,
            "max_temperature": max_temperature,
            "discharge_duration": discharge_duration
        }])


if raw_input_df is not None:
    prepared_df, error_message = prepare_uploaded_data(
        raw_input_df,
        feature_columns,
        training_df
    )

    if error_message:
        st.error(error_message)
        st.stop()

    model_input = prepared_df[feature_columns].copy()

    predictions = model.predict(model_input)

    result_df = raw_input_df.copy()
    result_df["predicted_soh_percent"] = predictions

    result_df["health_status"] = [
        (
            get_health_status(prediction)[0]
            if "Compatible" in compatibility
            else "Input Not Validated"
        )
        for prediction, compatibility in zip(
            predictions,
            prepared_df["input_compatibility"]
        )
    ]

    result_df["input_compatibility"] = prepared_df[
        "input_compatibility"
    ]

    st.subheader("Prediction Results")

    st.dataframe(
        result_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download Prediction Results",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="battery_soh_predictions.csv",
        mime="text/csv"
    )

    selected_row_index = st.selectbox(
        "Select a Result Row for Detailed Explanation",
        options=list(range(len(result_df))),
        format_func=lambda index: f"Uploaded Row {index + 1}"
    )

    selected_prediction = predictions[selected_row_index]

    compatibility_message = prepared_df[
        "input_compatibility"
    ].iloc[selected_row_index]

    is_compatible = "Compatible" in compatibility_message

    status, icon, message = get_health_status(selected_prediction)

    if not is_compatible:
        status = "Input Not Validated"

    metric_1, metric_2, metric_3 = st.columns(3)

    metric_1.metric(
        "Predicted SOH",
        f"{selected_prediction:.2f}%"
    )

    metric_2.metric(
        "Battery Status",
        status
    )

    metric_3.metric(
        "Input Compatibility",
        (
            "Within Range"
            if "Compatible" in prepared_df[
                "input_compatibility"
            ].iloc[selected_row_index]
            else "Review Required"
        )
    )

    if not is_compatible:
        st.error(
            "Input Not Validated: one or more feature values are outside "
            "the observed training-data range. This SOH prediction is shown "
            "for diagnostic purposes only and must not be used for a "
            "maintenance decision."
        )

    elif status == "Healthy":
        st.success(f"{icon} {message}")

    elif status == "Monitor":
        st.warning(f"{icon} {message}")

    else:
        st.error(f"{icon} {message}")

    compatibility_message = prepared_df[
        "input_compatibility"
    ].iloc[selected_row_index]

    if "Review input" in compatibility_message:
        st.warning(
            "Input-quality warning: "
            + compatibility_message
            + ". The prediction may be less reliable because this "
              "combination was not well represented in training data."
        )
    else:
        st.success(
            "Input-quality check: values are within the observed "
            "NASA training-data range."
        )

    st.subheader("Local Explainable AI")

    show_local_shap(
        model,
        model_input.iloc[[selected_row_index]]
    )

    st.caption(
        "This page is a software demonstration. For real-hardware use, "
        "the model must be calibrated and validated using sensor data "
        "from the actual battery cell/module."
    )