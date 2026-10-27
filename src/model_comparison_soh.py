"""
SOH model comparison using leave-one-battery-out validation.

Compares:
1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

The best model is selected using the lowest mean LOBO MAE and then
retrained on all four batteries for later use in the dashboard.
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from multi_battery import process_battery


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

# Each NASA battery used a different discharge cutoff voltage.
CUTOFF_VOLTAGES = {
    "B0005": 2.7,
    "B0006": 2.5,
    "B0007": 2.2,
    "B0018": 2.5
}

# cycle_number, battery_id, SOH, RUL and eol_cycle are deliberately excluded.
FEATURE_COLUMNS = [
    "mean_voltage",
    "voltage_headroom",
    "mean_current",
    "mean_temperature",
    "max_temperature",
    "discharge_duration"
]


def build_dataset():
    """Load all batteries and create one common SOH feature table."""
    battery_frames = []

    for battery_id in BATTERIES:
        file_path = DATASET_DIR / f"{battery_id}.mat"
        battery_frames.append(process_battery(file_path, battery_id))

    df = pd.concat(battery_frames, ignore_index=True)

    # Normalize minimum voltage with battery-specific cutoff voltage.
    df["cutoff_voltage"] = df["battery_id"].map(CUTOFF_VOLTAGES)

    df["voltage_headroom"] = (
        df["min_voltage"] - df["cutoff_voltage"]
    )

    return df.sort_values(
        ["battery_id", "cycle_number"]
    ).reset_index(drop=True)


def create_models():
    """
    Conservative model configurations.

    Dataset is small, so parameters are intentionally controlled
    to reduce unnecessary overfitting.
    """
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regression", LinearRegression())
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            max_depth=2,
            learning_rate=0.04,
            min_child_weight=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )
    }


def calculate_metrics(actual, predicted):
    """Return standard regression metrics."""
    return {
        "RMSE_percent": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE_percent": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted)
    }


def plot_fold_predictions(test_df, predictions_by_model, test_battery):
    """Create actual-vs-predicted SOH graph for one held-out battery."""

    colours = {
        "Linear Regression": "#f59e0b",
        "Random Forest": "#2563eb",
        "XGBoost": "#dc2626"
    }

    plt.figure(figsize=(10, 6))

    plt.plot(
        test_df["cycle_number"],
        test_df["SOH"],
        color="black",
        linewidth=2.5,
        label="Actual SOH"
    )

    for model_name, predictions in predictions_by_model.items():
        plt.plot(
            test_df["cycle_number"],
            predictions,
            linewidth=1.8,
            color=colours[model_name],
            label=model_name
        )

    plt.xlabel(f"Discharge Cycle Number ({test_battery})")
    plt.ylabel("SOH (%)")
    plt.title(f"SOH Prediction — Held-Out Battery: {test_battery}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_file = RESULTS_DIR / f"soh_model_comparison_{test_battery}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_summary(summary_df):
    """Create a compact chart comparing mean LOBO performance."""

    ordered_df = summary_df.sort_values("MAE_percent")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(
        ordered_df["model"],
        ordered_df["MAE_percent"],
        color="#2563eb"
    )
    axes[0].set_title("Mean LOBO MAE (Lower is Better)")
    axes[0].set_ylabel("MAE (Percentage Points)")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(
        ordered_df["model"],
        ordered_df["R2"],
        color="#16a34a"
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Mean LOBO R² (Higher is Better)")
    axes[1].set_ylabel("R²")
    axes[1].tick_params(axis="x", rotation=15)

    fig.suptitle(
        "SOH Model Comparison: Leave-One-Battery-Out Validation",
        y=1.03
    )

    fig.tight_layout()

    plt.savefig(
        RESULTS_DIR / "soh_model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------
    df = build_dataset()

    df.to_csv(
        RESULTS_DIR / "all_batteries_soh_features.csv",
        index=False
    )

    fold_results = []
    prediction_rows = []

    # ------------------------------------------------------------
    # 2. Leave-One-Battery-Out validation
    # ------------------------------------------------------------
    for test_battery in BATTERIES:

        train_df = df[df["battery_id"] != test_battery].copy()
        test_df = df[df["battery_id"] == test_battery].copy()

        print("\n" + "=" * 65)
        print(
            f"Testing on {test_battery}; "
            "training on the other three batteries"
        )
        print("=" * 65)

        fold_predictions = {}

        for model_name, model in create_models().items():

            # Train only on the three training batteries.
            model.fit(
                train_df[FEATURE_COLUMNS],
                train_df["SOH"]
            )

            predictions = model.predict(
                test_df[FEATURE_COLUMNS]
            )

            fold_predictions[model_name] = predictions

            fold_metrics = calculate_metrics(
                test_df["SOH"],
                predictions
            )

            fold_results.append({
                "model": model_name,
                "test_battery": test_battery,
                "test_rows": len(test_df),
                **fold_metrics
            })

            for cycle_number, actual, predicted in zip(
                test_df["cycle_number"],
                test_df["SOH"],
                predictions
            ):
                prediction_rows.append({
                    "model": model_name,
                    "test_battery": test_battery,
                    "cycle_number": cycle_number,
                    "actual_soh": actual,
                    "predicted_soh": predicted,
                    "absolute_error": abs(actual - predicted)
                })

            print(
                f"{model_name:<18} | "
                f"RMSE = {fold_metrics['RMSE_percent']:.3f}% | "
                f"MAE = {fold_metrics['MAE_percent']:.3f}% | "
                f"R² = {fold_metrics['R2']:.4f}"
            )

        plot_fold_predictions(
            test_df,
            fold_predictions,
            test_battery
        )

    # ------------------------------------------------------------
    # 3. Save fold-wise and average results
    # ------------------------------------------------------------
    fold_results_df = pd.DataFrame(fold_results)

    summary_df = (
        fold_results_df
        .groupby("model", as_index=False)[
            ["RMSE_percent", "MAE_percent", "R2"]
        ]
        .mean()
        .sort_values("MAE_percent")
        .reset_index(drop=True)
    )

    fold_results_df.to_csv(
        RESULTS_DIR / "soh_model_comparison_by_battery.csv",
        index=False
    )

    summary_df.to_csv(
        RESULTS_DIR / "soh_model_comparison.csv",
        index=False
    )

    predictions_df = pd.DataFrame(prediction_rows)

    predictions_df.to_csv(
        RESULTS_DIR / "soh_predictions_all_models.csv",
        index=False
    )

    plot_summary(summary_df)

    # ------------------------------------------------------------
    # 4. Retrain best LOBO model on all four batteries
    # ------------------------------------------------------------
    best_model_name = summary_df.iloc[0]["model"]

    best_model = create_models()[best_model_name]

    best_model.fit(
        df[FEATURE_COLUMNS],
        df["SOH"]
    )

    model_file = MODELS_DIR / "best_soh_model.joblib"

    joblib.dump(best_model, model_file)

    metadata = {
        "selected_model": best_model_name,
        "selection_rule": (
            "Lowest mean MAE across four "
            "leave-one-battery-out validation folds"
        ),
        "feature_columns": FEATURE_COLUMNS,
        "validation_method": (
            "LOBO validation using B0005, B0006, B0007 and B0018"
        ),
        "mean_metrics": (
            summary_df
            .set_index("model")
            .round(6)
            .to_dict(orient="index")
        )
    }

    metadata_file = MODELS_DIR / "best_soh_model_metadata.json"

    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # ------------------------------------------------------------
    # 5. Print final result
    # ------------------------------------------------------------
    print("\n\nFINAL MEAN LOBO RESULTS")
    print(summary_df.round(3).to_string(index=False))

    print(f"\nSelected best model: {best_model_name}")
    print("\nSaved files:")
    print("results/all_batteries_soh_features.csv")
    print("results/soh_model_comparison_by_battery.csv")
    print("results/soh_model_comparison.csv")
    print("results/soh_predictions_all_models.csv")
    print("results/soh_model_comparison.png")
    print("results/soh_model_comparison_B0005.png")
    print("results/soh_model_comparison_B0006.png")
    print("results/soh_model_comparison_B0007.png")
    print("results/soh_model_comparison_B0018.png")
    print("models/best_soh_model.joblib")
    print("models/best_soh_model_metadata.json")


if __name__ == "__main__":
    main()