"""
RUL model comparison using leave-one-battery-out validation.

Compares:
1. Linear Regression: snapshot features only
2. Linear Regression: snapshot + degradation-trend features
3. Random Forest: snapshot + degradation-trend features
4. XGBoost: snapshot + degradation-trend features

Important:
- Only pre-EOL cycles are used.
- RUL is never negative.
- cycle_number, battery_id, SOH and EOL cycle are NOT used as model features.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from multi_battery import process_battery, add_trend_features


# ------------------------------------------------------------
# Project paths and constants
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

CUTOFF_VOLTAGES = {
    "B0005": 2.7,
    "B0006": 2.5,
    "B0007": 2.2,
    "B0018": 2.5
}

SNAPSHOT_FEATURES = [
    "mean_voltage",
    "voltage_headroom",
    "mean_current",
    "mean_temperature",
    "max_temperature",
    "discharge_duration"
]

TREND_BASE_COLUMNS = [
    "mean_voltage",
    "voltage_headroom",
    "mean_temperature",
    "max_temperature",
    "discharge_duration"
]


def build_dataset():
    """Load all batteries, normalize voltage and add past-only trend features."""
    battery_frames = []

    for battery_id in BATTERIES:
        file_path = DATASET_DIR / f"{battery_id}.mat"
        battery_frames.append(process_battery(file_path, battery_id))

    df = pd.concat(battery_frames, ignore_index=True)

    df["cutoff_voltage"] = df["battery_id"].map(CUTOFF_VOLTAGES)

    df["voltage_headroom"] = (
        df["min_voltage"] - df["cutoff_voltage"]
    )

    # Adds delta, slope and variation features separately for every battery.
    df = add_trend_features(df)

    return df.sort_values(
        ["battery_id", "cycle_number"]
    ).reset_index(drop=True)


def get_trend_features():
    """Return all degradation-trend feature names."""
    trend_features = []

    for column in TREND_BASE_COLUMNS:
        trend_features.extend([
            f"{column}_delta_1",
            f"{column}_slope_5",
            f"{column}_slope_10",
            f"{column}_std_5"
        ])

    return trend_features


def create_models():
    """
    Use conservative settings because there are only four batteries.
    No cycle number or battery ID is used as an input feature.
    """
    trend_features = SNAPSHOT_FEATURES + get_trend_features()

    return {
        "Linear Regression (Snapshot)": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("regression", LinearRegression())
            ]),
            "features": SNAPSHOT_FEATURES
        },

        "Linear Regression (Trend)": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("regression", LinearRegression())
            ]),
            "features": trend_features
        },

        "Random Forest (Trend)": {
            "model": RandomForestRegressor(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=3,
                max_features=0.8,
                random_state=42,
                n_jobs=-1
            ),
            "features": trend_features
        },

        "XGBoost (Trend)": {
            "model": XGBRegressor(
                objective="reg:squarederror",
                n_estimators=250,
                max_depth=2,
                learning_rate=0.04,
                min_child_weight=4,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=8.0,
                random_state=42,
                n_jobs=-1,
                tree_method="hist"
            ),
            "features": trend_features
        }
    }


def calculate_metrics(actual, predicted):
    """Calculate standard RUL regression metrics."""
    return {
        "RMSE_cycles": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE_cycles": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted)
    }


def plot_fold_predictions(test_df, predictions_by_model, test_battery):
    """Create actual-vs-predicted RUL graph for one unseen battery."""

    colours = {
        "Linear Regression (Snapshot)": "#f59e0b",
        "Linear Regression (Trend)": "#dc2626",
        "Random Forest (Trend)": "#2563eb",
        "XGBoost (Trend)": "#7c3aed"
    }

    plt.figure(figsize=(11, 6))

    plt.plot(
        test_df["cycle_number"],
        test_df["RUL"],
        color="black",
        linewidth=2.5,
        label="Actual RUL"
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
    plt.ylabel("RUL (cycles)")
    plt.title(f"RUL Prediction — Held-Out Battery: {test_battery}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"rul_model_comparison_{test_battery}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_summary(summary_df):
    """Create final average-LOBO comparison chart."""

    ordered_df = summary_df.sort_values("MAE_cycles")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        ordered_df["model"],
        ordered_df["MAE_cycles"],
        color="#2563eb"
    )

    axes[0].set_title("Mean LOBO MAE (Lower is Better)")
    axes[0].set_ylabel("MAE (cycles)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(
        ordered_df["model"],
        ordered_df["R2"],
        color="#16a34a"
    )

    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Mean LOBO R² (Higher is Better)")
    axes[1].set_ylabel("R²")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle(
        "RUL Model Comparison: Leave-One-Battery-Out Validation",
        y=1.03
    )

    fig.tight_layout()

    plt.savefig(
        RESULTS_DIR / "rul_model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------
    # 1. Build complete feature dataset
    # ------------------------------------------------------------
    all_df = build_dataset()

    trend_features = get_trend_features()
    all_rul_features = SNAPSHOT_FEATURES + trend_features

    # Use only cycles up to that battery's 80% SOH end-of-life cycle.
    pre_eol_df = all_df[
        all_df["cycle_number"] <= all_df["eol_cycle"]
    ].copy()

    # The first nine cycles cannot have a 10-cycle slope/history.
    valid_rul_df = pre_eol_df.dropna(
        subset=all_rul_features
    ).reset_index(drop=True)

    valid_rul_df.to_csv(
        RESULTS_DIR / "all_batteries_rul_features.csv",
        index=False
    )

    print("\nRUL rows after removing early cycles without trend history:")
    print(valid_rul_df.groupby("battery_id").size())

    # ------------------------------------------------------------
    # 2. Leave-One-Battery-Out evaluation
    # ------------------------------------------------------------
    fold_results = []
    prediction_rows = []

    for test_battery in BATTERIES:

        train_df = valid_rul_df[
            valid_rul_df["battery_id"] != test_battery
        ].copy()

        test_df = valid_rul_df[
            valid_rul_df["battery_id"] == test_battery
        ].copy()

        print("\n" + "=" * 70)
        print(
            f"Testing RUL on {test_battery}; "
            "training on the other three batteries"
        )
        print("=" * 70)

        predictions_by_model = {}

        for model_name, model_info in create_models().items():

            model = model_info["model"]
            feature_columns = model_info["features"]

            model.fit(
                train_df[feature_columns],
                train_df["RUL"]
            )

            predictions = model.predict(
                test_df[feature_columns]
            )

            # Physical constraint: remaining life cannot be negative.
            predictions = np.maximum(predictions, 0)

            predictions_by_model[model_name] = predictions

            fold_metrics = calculate_metrics(
                test_df["RUL"],
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
                test_df["RUL"],
                predictions
            ):
                prediction_rows.append({
                    "model": model_name,
                    "test_battery": test_battery,
                    "cycle_number": cycle_number,
                    "actual_rul": actual,
                    "predicted_rul": predicted,
                    "absolute_error_cycles": abs(actual - predicted)
                })

            print(
                f"{model_name:<30} | "
                f"RMSE = {fold_metrics['RMSE_cycles']:.3f} cycles | "
                f"MAE = {fold_metrics['MAE_cycles']:.3f} cycles | "
                f"R² = {fold_metrics['R2']:.4f}"
            )

        plot_fold_predictions(
            test_df,
            predictions_by_model,
            test_battery
        )

    # ------------------------------------------------------------
    # 3. Save results
    # ------------------------------------------------------------
    fold_results_df = pd.DataFrame(fold_results)

    summary_df = (
        fold_results_df
        .groupby("model", as_index=False)[
            ["RMSE_cycles", "MAE_cycles", "R2"]
        ]
        .mean()
        .sort_values("MAE_cycles")
        .reset_index(drop=True)
    )

    fold_results_df.to_csv(
        RESULTS_DIR / "rul_model_comparison_by_battery.csv",
        index=False
    )

    summary_df.to_csv(
        RESULTS_DIR / "rul_model_comparison.csv",
        index=False
    )

    pd.DataFrame(prediction_rows).to_csv(
        RESULTS_DIR / "rul_predictions_all_models.csv",
        index=False
    )

    plot_summary(summary_df)

    # ------------------------------------------------------------
    # 4. Final output
    # ------------------------------------------------------------
    print("\n\nFINAL MEAN LOBO RUL RESULTS")
    print(summary_df.round(3).to_string(index=False))

    print(
        "\nImportant: Select an RUL model only after checking "
        "both the average results and each held-out battery result."
    )

    print("\nSaved files:")
    print("results/all_batteries_rul_features.csv")
    print("results/rul_model_comparison_by_battery.csv")
    print("results/rul_model_comparison.csv")
    print("results/rul_predictions_all_models.csv")
    print("results/rul_model_comparison.png")

    for battery_id in BATTERIES:
        print(f"results/rul_model_comparison_{battery_id}.png")


if __name__ == "__main__":
    main()