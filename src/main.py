import os
os.makedirs("results", exist_ok=True)
import pandas as pd
from data_loader import load_battery_data
from soh_rul import calculate_soh, find_eol_cycle, calculate_rul, clip_rul
import matplotlib.pyplot as plt
from scipy.io import loadmat
from feature_extraction import extract_features
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

caps_6, cycles_6, cycle_data_6 = load_battery_data("dataset/B0006.mat", "B0006")
print("B0006 total discharge cycles:", len(caps_6))
print("First capacity:", caps_6[0])
print("Last capacity:", caps_6[-1])

capacities, cycle_numbers, cycle_data_unused = load_battery_data("dataset/B0005.mat", "B0005")

print(f"\nTotal discharge cycles extracted: {len(capacities)}")
print("First 5 capacities:", capacities[:5])
print("Last 5 capacities:", capacities[-5:])

plt.plot(cycle_numbers, capacities, marker='o', markersize=3)
plt.xlabel("Discharge Cycle Number")
plt.ylabel("Capacity (Ah)")
plt.title("B0005 - Capacity Fade over Cycles")
plt.grid(True)
plt.savefig("results/capacity_vs_cycle.png", dpi=300, bbox_inches='tight')
plt.show()

# --- calculate SOH ---
soh, initial_capacity = calculate_soh(capacities)

print(f"\nInitial capacity (100% baseline): {initial_capacity:.4f} Ah")
print("First 5 SOH values:", [round(s, 2) for s in soh[:5]])
print("Last 5 SOH values:", [round(s, 2) for s in soh[-5:]])

plt.figure()
plt.plot(cycle_numbers, soh, marker='o', markersize=3, color='green')
plt.xlabel("Discharge Cycle Number")
plt.ylabel("SOH (%)")
plt.title("B0005 - State of Health (SOH) over Cycles")
plt.grid(True)
plt.savefig("results/soh_vs_cycle.png", dpi=300, bbox_inches='tight')
plt.show()

# --- find End-of-Life cycle ---
eol_cycle = find_eol_cycle(cycle_numbers, soh, eol_threshold=80.0)

print(f"\nEOL threshold: 80.0% SOH")
print(f"First cycle reaching EOL: {eol_cycle}")

# --- compute RUL ---
rul = calculate_rul(cycle_numbers, eol_cycle)

print("First 5 RUL values:", rul[:5])
print("RUL at/near EOL cycle:", rul[eol_cycle-5:eol_cycle+3])

plt.figure()
plt.plot(cycle_numbers, rul, marker='o', markersize=3, color='red')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Discharge Cycle Number")
plt.ylabel("RUL (cycles)")
plt.title("B0005 - Remaining Useful Life (RUL)")
plt.grid(True)
plt.savefig("results/rul_vs_cycle.png", dpi=300, bbox_inches='tight')
plt.show()

# --- clip negative RUL ---
rul_clipped = clip_rul(rul)

print("RUL before clipping (last 10):", rul[-10:])
print("RUL after clipping (last 10):", rul_clipped[-10:])

# temporary raw inspection
raw = loadmat("dataset/B0005.mat")
cycle_data_raw = raw["B0005"]["cycle"][0, 0][0]

types = [cycle_data_raw[i]["type"][0] for i in range(cycle_data_raw.shape[0])]
first_discharge_idx = types.index("discharge")

sample = cycle_data_raw[first_discharge_idx]["data"][0, 0]

voltage = sample["Voltage_measured"][0]
current = sample["Current_measured"][0]
temperature = sample["Temperature_measured"][0]
time = sample["Time"][0]

print("Voltage array shape:", voltage.shape)
print("Current array shape:", current.shape)
print("Temperature array shape:", temperature.shape)
print("Time array shape:", time.shape)

print("\nFirst 5 voltage readings:", voltage[:5])
print("First 5 current readings:", current[:5])
print("First 5 temperature readings:", temperature[:5])
print("First 5 time values (seconds):", time[:5])

# test on first discharge cycle only
test_features = extract_features(cycle_data_raw[first_discharge_idx])
print("\nFeatures for first discharge cycle:")
print(test_features)

# --- extract features for ALL discharge cycles ---
all_features = []

for i in range(cycle_data_raw.shape[0]):
    if cycle_data_raw[i]["type"][0] == "discharge":
        feat = extract_features(cycle_data_raw[i])
        all_features.append(feat)

print(f"\nTotal feature sets extracted: {len(all_features)}")
print("First feature set:", all_features[0])
print("Last feature set:", all_features[-1])

# combine all features + soh + rul into one DataFrame
df = pd.DataFrame(all_features)
df["cycle_number"] = cycle_numbers
df["SOH"] = soh
df["RUL"] = rul_clipped

print(df.head())
print("\nDataFrame shape:", df.shape)

df.to_csv("results/battery_features.csv", index=False)
print("\nSaved to results/battery_features.csv")

# --- train/test split (cycle-ordered, not random) for SOH ---
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print(f"\nTotal rows: {len(df)}")
print(f"Train rows: {len(train_df)} (cycles {train_df['cycle_number'].min()} to {train_df['cycle_number'].max()})")
print(f"Test rows: {len(test_df)} (cycles {test_df['cycle_number'].min()} to {test_df['cycle_number'].max()})")

feature_columns = ["mean_voltage", "min_voltage", "mean_current", "mean_temperature", "max_temperature", "discharge_duration"]

X_train = train_df[feature_columns]
y_train = train_df["SOH"]

X_test = test_df[feature_columns]
y_test = test_df["SOH"]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# --- train baseline SOH model ---
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- SOH Model Evaluation ---")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R² Score: {r2:.4f}")

print("\nActual vs Predicted SOH (first 10 test cycles):")
for actual, pred in zip(y_test.values[:10], y_pred[:10]):
    print(f"Actual: {actual:.2f}   Predicted: {pred:.2f}")

plt.figure()
plt.plot(test_df["cycle_number"], y_test.values, marker='o', markersize=4, label="Actual SOH", color='green')
plt.plot(test_df["cycle_number"], y_pred, marker='x', markersize=4, label="Predicted SOH", color='orange')
plt.xlabel("Discharge Cycle Number")
plt.ylabel("SOH (%)")
plt.title("B0005 - Actual vs Predicted SOH (Test Set)")
plt.legend()
plt.grid(True)
plt.savefig("results/actual_vs_predicted_soh.png", dpi=300, bbox_inches='tight')
plt.show()

# --- RUL prediction: restrict to pre-EOL cycles only (RUL is meaningless after EOL) ---
rul_df = df[df["cycle_number"] <= eol_cycle].reset_index(drop=True)

rul_split_index = int(len(rul_df) * 0.8)
rul_train_df = rul_df.iloc[:rul_split_index]
rul_test_df = rul_df.iloc[rul_split_index:]

print(f"\nRUL modeling dataset: {len(rul_df)} rows (cycles 1 to {eol_cycle})")
print(f"RUL train rows: {len(rul_train_df)} (cycles {rul_train_df['cycle_number'].min()} to {rul_train_df['cycle_number'].max()})")
print(f"RUL test rows: {len(rul_test_df)} (cycles {rul_test_df['cycle_number'].min()} to {rul_test_df['cycle_number'].max()})")

X_train_rul = rul_train_df[feature_columns]
y_train_rul = rul_train_df["RUL"]

X_test_rul = rul_test_df[feature_columns]
y_test_rul = rul_test_df["RUL"]

model_rul = LinearRegression()
model_rul.fit(X_train_rul, y_train_rul)

y_pred_rul = model_rul.predict(X_test_rul)

rmse_rul = np.sqrt(mean_squared_error(y_test_rul, y_pred_rul))
mae_rul = mean_absolute_error(y_test_rul, y_pred_rul)
r2_rul = r2_score(y_test_rul, y_pred_rul)

print(f"\n--- RUL Model Evaluation (pre-EOL cycles only) ---")
print(f"RMSE: {rmse_rul:.3f}")
print(f"MAE: {mae_rul:.3f}")
print(f"R² Score: {r2_rul:.4f}")

print("\nActual vs Predicted RUL (first 10 test cycles):")
for actual, pred in zip(y_test_rul.values[:10], y_pred_rul[:10]):
    print(f"Actual: {actual:.1f}   Predicted: {pred:.1f}")

plt.figure()
plt.plot(rul_test_df["cycle_number"], y_test_rul.values, marker='o', markersize=4, label="Actual RUL", color='blue')
plt.plot(rul_test_df["cycle_number"], y_pred_rul, marker='x', markersize=4, label="Predicted RUL", color='red')
plt.xlabel("Discharge Cycle Number")
plt.ylabel("RUL (cycles)")
plt.title("B0005 - Actual vs Predicted RUL (Pre-EOL Test Set)")
plt.legend()
plt.grid(True)
plt.savefig("results/actual_vs_predicted_rul.png", dpi=300, bbox_inches='tight')
plt.show()

from sklearn.ensemble import RandomForestRegressor

# --- Random Forest for RUL (comparison against Linear Regression) ---
model_rul_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rul_rf.fit(X_train_rul, y_train_rul)

y_pred_rul_rf = model_rul_rf.predict(X_test_rul)

rmse_rul_rf = np.sqrt(mean_squared_error(y_test_rul, y_pred_rul_rf))
mae_rul_rf = mean_absolute_error(y_test_rul, y_pred_rul_rf)
r2_rul_rf = r2_score(y_test_rul, y_pred_rul_rf)

print(f"\n--- RUL Model Evaluation: Random Forest ---")
print(f"RMSE: {rmse_rul_rf:.3f}")
print(f"MAE: {mae_rul_rf:.3f}")
print(f"R² Score: {r2_rul_rf:.4f}")

print("\nActual vs Predicted RUL - Random Forest (first 10 test cycles):")
for actual, pred in zip(y_test_rul.values[:10], y_pred_rul_rf[:10]):
    print(f"Actual: {actual:.1f}   Predicted: {pred:.1f}")

plt.figure()
plt.plot(rul_test_df["cycle_number"], y_test_rul.values, marker='o', markersize=4, label="Actual RUL", color='blue')
plt.plot(rul_test_df["cycle_number"], y_pred_rul, marker='x', markersize=4, label="Predicted RUL (Linear Regression)", color='red')
plt.plot(rul_test_df["cycle_number"], y_pred_rul_rf, marker='s', markersize=4, label="Predicted RUL (Random Forest)", color='purple')
plt.xlabel("Discharge Cycle Number")
plt.ylabel("RUL (cycles)")
plt.title("B0005 - RUL: Linear Regression vs Random Forest")
plt.legend()
plt.grid(True)
plt.savefig("results/rul_model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()