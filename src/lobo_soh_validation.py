import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from multi_battery import process_battery


def get_metrics(actual, predicted):
    return {
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted)
    }


def evaluate_soh_fold(df, test_battery, feature_columns):
    """
    Train SOH model on three batteries and test on one completely unseen battery.
    """
    train_df = df[df["battery_id"] != test_battery].copy()
    test_df = df[df["battery_id"] == test_battery].copy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression())
    ])

    model.fit(
        train_df[feature_columns],
        train_df["SOH"]
    )

    predictions = model.predict(
        test_df[feature_columns]
    )

    metrics = get_metrics(test_df["SOH"], predictions)

    # Plot SOH prediction for this held-out battery
    plt.figure(figsize=(10, 6))

    plt.plot(
        test_df["cycle_number"],
        test_df["SOH"],
        marker="o",
        markersize=4,
        linewidth=2,
        color="green",
        label="Actual SOH"
    )

    plt.plot(
        test_df["cycle_number"],
        predictions,
        marker="x",
        markersize=5,
        linewidth=1.8,
        color="orange",
        label="Predicted SOH"
    )

    plt.xlabel(f"Discharge Cycle Number ({test_battery})")
    plt.ylabel("SOH (%)")
    plt.title(f"SOH Prediction — Held-Out Battery: {test_battery}")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"results/lobo_soh_{test_battery}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "test_battery": test_battery,
        "test_rows": len(test_df),
        "RMSE_percent": metrics["RMSE"],
        "MAE_percent": metrics["MAE"],
        "R2": metrics["R2"]
    }


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load all batteries
    # ------------------------------------------------------------
    df_b5 = process_battery("dataset/B0005.mat", "B0005")
    df_b6 = process_battery("dataset/B0006.mat", "B0006")
    df_b7 = process_battery("dataset/B0007.mat", "B0007")
    df_b18 = process_battery("dataset/B0018.mat", "B0018")

    all_df = pd.concat(
        [df_b5, df_b6, df_b7, df_b18],
        ignore_index=True
    )

    # ------------------------------------------------------------
    # 2. Create voltage-headroom feature
    # ------------------------------------------------------------
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

    # Only physical per-cycle features.
    # Do NOT include cycle number, SOH, RUL, EOL cycle, or battery ID.
    feature_columns = [
        "mean_voltage",
        "voltage_headroom",
        "mean_current",
        "mean_temperature",
        "max_temperature",
        "discharge_duration"
    ]

    # ------------------------------------------------------------
    # 3. Leave-one-battery-out validation
    # ------------------------------------------------------------
    batteries = ["B0005", "B0006", "B0007", "B0018"]
    results = []

    for test_battery in batteries:
        print(f"\n{'=' * 60}")
        print(f"Testing SOH on {test_battery}; training on other 3 batteries")
        print(f"{'=' * 60}")

        fold_result = evaluate_soh_fold(
            all_df,
            test_battery,
            feature_columns
        )

        results.append(fold_result)

        print(
            f"RMSE = {fold_result['RMSE_percent']:.3f} %, "
            f"MAE = {fold_result['MAE_percent']:.3f} %, "
            f"R² = {fold_result['R2']:.4f}"
        )

    # ------------------------------------------------------------
    # 4. Save final results table
    # ------------------------------------------------------------
    results_df = pd.DataFrame(results)

    average_row = {
        "test_battery": "Average",
        "test_rows": results_df["test_rows"].mean(),
        "RMSE_percent": results_df["RMSE_percent"].mean(),
        "MAE_percent": results_df["MAE_percent"].mean(),
        "R2": results_df["R2"].mean()
    }

    results_df = pd.concat(
        [results_df, pd.DataFrame([average_row])],
        ignore_index=True
    )

    results_df.to_csv(
        "results/lobo_soh_validation_results.csv",
        index=False
    )

    print("\n\nFINAL SOH LEAVE-ONE-BATTERY-OUT RESULTS")
    print(results_df.round(3).to_string(index=False))

    print("\nSaved:")
    print("results/lobo_soh_validation_results.csv")
    print("results/lobo_soh_B0005.png")
    print("results/lobo_soh_B0006.png")
    print("results/lobo_soh_B0007.png")
    print("results/lobo_soh_B0018.png")