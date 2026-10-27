import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_battery_data
from soh_rul import calculate_soh, find_eol_cycle, calculate_rul, clip_rul
from feature_extraction import extract_features

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def process_battery(file_path, battery_key, eol_threshold=80.0):
    """Create one feature row for every discharge cycle of a battery."""
    capacities, cycle_numbers, cycle_data = load_battery_data(file_path, battery_key)

    soh, initial_capacity = calculate_soh(capacities)
    eol_cycle = find_eol_cycle(cycle_numbers, soh, eol_threshold)
    rul = calculate_rul(cycle_numbers, eol_cycle)
    rul_clipped = clip_rul(rul)

    all_features = []

    for i in range(cycle_data.shape[0]):
        if cycle_data[i]["type"][0] == "discharge":
            all_features.append(extract_features(cycle_data[i]))

    df = pd.DataFrame(all_features)
    df["cycle_number"] = cycle_numbers
    df["SOH"] = soh
    df["RUL"] = rul_clipped
    df["battery_id"] = battery_key
    df["eol_cycle"] = eol_cycle

    print(
        f"{battery_key}: {len(df)} discharge cycles | "
        f"EOL cycle: {eol_cycle} | "
        f"Initial capacity: {initial_capacity:.4f} Ah"
    )

    return df


def rolling_slope(values):
    """Calculate the linear slope across a rolling window."""
    x = np.arange(len(values))
    return np.polyfit(x, values, 1)[0]


def add_trend_features(df):
    """
    Add degradation-trend features using only the current and earlier cycles.
    Each battery is processed separately to avoid mixing histories.
    """
    df = df.sort_values(["battery_id", "cycle_number"]).copy()

    trend_base_columns = [
        "mean_voltage",
        "voltage_headroom",
        "mean_temperature",
        "max_temperature",
        "discharge_duration"
    ]

    grouped = df.groupby("battery_id")

    for col in trend_base_columns:
        # Difference from the immediately previous completed cycle
        df[f"{col}_delta_1"] = grouped[col].diff(1)

        # True recent linear trend over 5 cycles
        df[f"{col}_slope_5"] = grouped[col].transform(
            lambda s: s.rolling(window=5, min_periods=5).apply(
                rolling_slope,
                raw=True
            )
        )

        # Longer-term linear trend over 10 cycles
        df[f"{col}_slope_10"] = grouped[col].transform(
            lambda s: s.rolling(window=10, min_periods=10).apply(
                rolling_slope,
                raw=True
            )
        )

        # Short-term variation / instability over 5 cycles
        df[f"{col}_std_5"] = grouped[col].transform(
            lambda s: s.rolling(window=5, min_periods=5).std()
        )

    return df


def print_metrics(model_name, actual, predicted, unit):
    """Print standard regression metrics with the correct unit."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print(f"\n--- {model_name} ---")
    print(f"RMSE: {rmse:.3f} {unit}")
    print(f"MAE:  {mae:.3f} {unit}")
    print(f"R²:   {r2:.4f}")

    return rmse, mae, r2


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load and process all batteries
    # ------------------------------------------------------------
    df_b5 = process_battery("dataset/B0005.mat", "B0005")
    df_b6 = process_battery("dataset/B0006.mat", "B0006")
    df_b7 = process_battery("dataset/B0007.mat", "B0007")
    df_b18 = process_battery("dataset/B0018.mat", "B0018")

    all_batteries_df = pd.concat(
        [df_b5, df_b6, df_b7, df_b18],
        ignore_index=True
    )

    # ------------------------------------------------------------
    # 2. Normalize min voltage relative to cutoff voltage
    # ------------------------------------------------------------
    cutoff_voltages = {
        "B0005": 2.7,
        "B0006": 2.5,
        "B0007": 2.2,
        "B0018": 2.5
    }

    all_batteries_df["cutoff_voltage"] = (
        all_batteries_df["battery_id"].map(cutoff_voltages)
    )

    all_batteries_df["voltage_headroom"] = (
        all_batteries_df["min_voltage"]
        - all_batteries_df["cutoff_voltage"]
    )

    # ------------------------------------------------------------
    # 3. Add past-cycle degradation-trend features
    # ------------------------------------------------------------
    all_batteries_df = add_trend_features(all_batteries_df)

    all_batteries_df.to_csv(
        "results/all_batteries_features_with_trends.csv",
        index=False
    )

    print("\nSaved feature dataset:")
    print("results/all_batteries_features_with_trends.csv")

    # ------------------------------------------------------------
    # 4. Feature lists
    # ------------------------------------------------------------
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

    trend_features = []

    for col in trend_base_columns:
        trend_features.extend([
            f"{col}_delta_1",
            f"{col}_slope_5",
            f"{col}_slope_10",
            f"{col}_std_5"
        ])

    rul_features_with_trends = snapshot_features + trend_features

    # ------------------------------------------------------------
    # 5. SOH: train on B0005/6/7, test on unseen B0018
    # ------------------------------------------------------------
    soh_train_df = all_batteries_df[
        all_batteries_df["battery_id"] != "B0018"
    ].copy()

    soh_test_df = all_batteries_df[
        all_batteries_df["battery_id"] == "B0018"
    ].copy()

    soh_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression())
    ])

    soh_model.fit(
        soh_train_df[snapshot_features],
        soh_train_df["SOH"]
    )

    soh_predictions = soh_model.predict(
        soh_test_df[snapshot_features]
    )

    print_metrics(
        "SOH Model: B0005/B0006/B0007 -> B0018",
        soh_test_df["SOH"],
        soh_predictions,
        "%"
    )

    # ------------------------------------------------------------
    # 6. RUL: retain only cycles up to EOL
    # ------------------------------------------------------------
    pre_eol_df = all_batteries_df[
        all_batteries_df["cycle_number"]
        <= all_batteries_df["eol_cycle"]
    ].copy()

    # Rows before cycle 10 lack the full 10-cycle history.
    valid_rul_df = pre_eol_df.dropna(
        subset=rul_features_with_trends
    ).reset_index(drop=True)

    rul_train_df = valid_rul_df[
        valid_rul_df["battery_id"] != "B0018"
    ].copy()

    rul_test_df = valid_rul_df[
        valid_rul_df["battery_id"] == "B0018"
    ].copy()

    print(f"\nRUL training rows: {len(rul_train_df)}")
    print(f"RUL B0018 test rows: {len(rul_test_df)}")
    print(
        "B0018 test-cycle range:",
        f"{rul_test_df['cycle_number'].min()} to "
        f"{rul_test_df['cycle_number'].max()}"
    )

    # ------------------------------------------------------------
    # 7. Baseline RUL model: snapshot features only
    # ------------------------------------------------------------
    rul_snapshot_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression())
    ])

    rul_snapshot_model.fit(
        rul_train_df[snapshot_features],
        rul_train_df["RUL"]
    )

    rul_snapshot_predictions = rul_snapshot_model.predict(
        rul_test_df[snapshot_features]
    )

    # Physical constraint: RUL cannot be negative.
    rul_snapshot_predictions = np.maximum(rul_snapshot_predictions, 0)

    print_metrics(
        "RUL Baseline: Snapshot Features Only",
        rul_test_df["RUL"],
        rul_snapshot_predictions,
        "cycles"
    )

    # ------------------------------------------------------------
    # 8. Enhanced RUL model: snapshot + trend features
    # ------------------------------------------------------------
    rul_trend_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression())
    ])

    rul_trend_model.fit(
        rul_train_df[rul_features_with_trends],
        rul_train_df["RUL"]
    )

    rul_trend_predictions = rul_trend_model.predict(
        rul_test_df[rul_features_with_trends]
    )

    # Physical constraint: RUL cannot be negative.
    rul_trend_predictions = np.maximum(rul_trend_predictions, 0)

    print_metrics(
        "RUL Enhanced Model: Snapshot + Trend Features",
        rul_test_df["RUL"],
        rul_trend_predictions,
        "cycles"
    )

    # ------------------------------------------------------------
    # 9. Plot comparison
    # ------------------------------------------------------------
    plt.figure(figsize=(11, 7))

    plt.plot(
        rul_test_df["cycle_number"],
        rul_test_df["RUL"],
        marker="o",
        markersize=4,
        linewidth=2,
        color="blue",
        label="Actual RUL"
    )

    plt.plot(
        rul_test_df["cycle_number"],
        rul_snapshot_predictions,
        marker="x",
        markersize=5,
        linewidth=1.8,
        color="orange",
        label="Predicted RUL: Snapshot Only"
    )

    plt.plot(
        rul_test_df["cycle_number"],
        rul_trend_predictions,
        marker="x",
        markersize=5,
        linewidth=1.8,
        color="red",
        label="Predicted RUL: Snapshot + Trends"
    )

    plt.xlabel("Discharge Cycle Number (B0018)")
    plt.ylabel("RUL (cycles)")
    plt.title("B0018 RUL Prediction: Baseline vs Trend-Feature Model")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/rul_baseline_vs_trend_features.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()