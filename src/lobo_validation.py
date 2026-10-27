import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Reuse your existing functions from multi_battery.py
from multi_battery import process_battery, add_trend_features


def get_metrics(actual, predicted):
    return {
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted)
    }


def evaluate_one_fold(df, test_battery, snapshot_features, trend_features):
    """
    Train on three batteries and test on the one held-out battery.
    """
    train_df = df[df["battery_id"] != test_battery].copy()
    test_df = df[df["battery_id"] == test_battery].copy()

    # Snapshot-only baseline
    baseline_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression())
    ])

    baseline_model.fit(
        train_df[snapshot_features],
        train_df["RUL"]
    )

    baseline_predictions = baseline_model.predict(
        test_df[snapshot_features]
    )

    # RUL can never be negative
    baseline_predictions = np.maximum(baseline_predictions, 0)

    baseline_metrics = get_metrics(
        test_df["RUL"],
        baseline_predictions
    )

    # Snapshot + degradation-trend features
    trend_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression())
    ])

    trend_model.fit(
        train_df[trend_features],
        train_df["RUL"]
    )

    trend_predictions = trend_model.predict(
        test_df[trend_features]
    )

    # RUL can never be negative
    trend_predictions = np.maximum(trend_predictions, 0)

    trend_metrics = get_metrics(
        test_df["RUL"],
        trend_predictions
    )

    # Plot this held-out battery
    plt.figure(figsize=(10, 6))

    plt.plot(
        test_df["cycle_number"],
        test_df["RUL"],
        marker="o",
        markersize=4,
        color="blue",
        label="Actual RUL"
    )

    plt.plot(
        test_df["cycle_number"],
        baseline_predictions,
        marker="x",
        markersize=5,
        color="orange",
        label="Snapshot-only prediction"
    )

    plt.plot(
        test_df["cycle_number"],
        trend_predictions,
        marker="x",
        markersize=5,
        color="red",
        label="Snapshot + trend prediction"
    )

    plt.xlabel(f"Discharge Cycle Number ({test_battery})")
    plt.ylabel("RUL (cycles)")
    plt.title(f"RUL Prediction — Held-Out Battery: {test_battery}")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"results/lobo_rul_{test_battery}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "test_battery": test_battery,
        "test_rows": len(test_df),

        "baseline_RMSE": baseline_metrics["RMSE"],
        "baseline_MAE": baseline_metrics["MAE"],
        "baseline_R2": baseline_metrics["R2"],

        "trend_RMSE": trend_metrics["RMSE"],
        "trend_MAE": trend_metrics["MAE"],
        "trend_R2": trend_metrics["R2"]
    }


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------
    # Load all four batteries
    # ------------------------------------------------------------
    df_b5 = process_battery("dataset/B0005.mat", "B0005")
    df_b6 = process_battery("dataset/B0006.mat", "B0006")
    df_b7 = process_battery("dataset/B0007.mat", "B0007")
    df_b18 = process_battery("dataset/B0018.mat", "B0018")

    all_df = pd.concat(
        [df_b5, df_b6, df_b7, df_b18],
        ignore_index=True
    )

    # Battery-specific discharge cutoff voltages
    cutoff_voltages = {
        "B0005": 2.7,
        "B0006": 2.5,
        "B0007": 2.2,
        "B0018": 2.5
    }

    all_df["cutoff_voltage"] = all_df["battery_id"].map(cutoff_voltages)

    all_df["voltage_headroom"] = (
        all_df["min_voltage"] - all_df["cutoff_voltage"]
    )

    # Add past-only degradation features
    all_df = add_trend_features(all_df)

    # Use only pre-EOL cycles for RUL modelling
    pre_eol_df = all_df[
        all_df["cycle_number"] <= all_df["eol_cycle"]
    ].copy()

    snapshot_features = [
        "mean_voltage",
        "voltage_headroom",
        "mean_current",
        "mean_temperature",
        "max_temperature",
        "discharge_duration"
    ]

    trend_base_columns = [
        "mean_voltage",
        "voltage_headroom",
        "mean_temperature",
        "max_temperature",
        "discharge_duration"
    ]

    trend_only_features = []

    for col in trend_base_columns:
        trend_only_features.extend([
            f"{col}_delta_1",
            f"{col}_slope_5",
            f"{col}_slope_10",
            f"{col}_std_5"
        ])

    all_rul_features = snapshot_features + trend_only_features

    # Remove early cycles without 10-cycle trend history
    valid_rul_df = pre_eol_df.dropna(
        subset=all_rul_features
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Leave-one-battery-out validation
    # ------------------------------------------------------------
    batteries = ["B0005", "B0006", "B0007", "B0018"]
    results = []

    for test_battery in batteries:
        print(f"\n{'=' * 60}")
        print(f"Testing on {test_battery}; training on other 3 batteries")
        print(f"{'=' * 60}")

        fold_result = evaluate_one_fold(
            valid_rul_df,
            test_battery,
            snapshot_features,
            all_rul_features
        )

        results.append(fold_result)

        print(
            f"Baseline  | "
            f"RMSE={fold_result['baseline_RMSE']:.3f}, "
            f"MAE={fold_result['baseline_MAE']:.3f}, "
            f"R²={fold_result['baseline_R2']:.4f}"
        )

        print(
            f"Trend model | "
            f"RMSE={fold_result['trend_RMSE']:.3f}, "
            f"MAE={fold_result['trend_MAE']:.3f}, "
            f"R²={fold_result['trend_R2']:.4f}"
        )

    # ------------------------------------------------------------
    # Save and print final table
    # ------------------------------------------------------------
    results_df = pd.DataFrame(results)

    mean_row = {
        "test_battery": "Average",
        "test_rows": results_df["test_rows"].mean(),
        "baseline_RMSE": results_df["baseline_RMSE"].mean(),
        "baseline_MAE": results_df["baseline_MAE"].mean(),
        "baseline_R2": results_df["baseline_R2"].mean(),
        "trend_RMSE": results_df["trend_RMSE"].mean(),
        "trend_MAE": results_df["trend_MAE"].mean(),
        "trend_R2": results_df["trend_R2"].mean()
    }

    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row])],
        ignore_index=True
    )

    results_df.to_csv(
        "results/lobo_validation_results.csv",
        index=False
    )

    print("\n\nFINAL LEAVE-ONE-BATTERY-OUT RESULTS")
    print(results_df.round(3).to_string(index=False))

    print("\nSaved:")
    print("results/lobo_validation_results.csv")
    print("results/lobo_rul_B0005.png")
    print("results/lobo_rul_B0006.png")
    print("results/lobo_rul_B0007.png")
    print("results/lobo_rul_B0018.png")