from scipy.io import loadmat
import numpy as np

raw = loadmat("dataset/B0005.mat")
cycle_data = raw["B0005"]["cycle"][0, 0][0]

re_values = []
rct_values = []
cycle_index = []

impedance_count = 0
for i in range(cycle_data.shape[0]):
    if cycle_data[i]["type"][0] == "impedance":
        impedance_count += 1
        data = cycle_data[i]["data"][0, 0]
        re_val = data["Re"][0][0]
        rct_val = data["Rct"][0][0]
        re_values.append(re_val)
        rct_values.append(rct_val)
        cycle_index.append(impedance_count)

print(f"Total impedance measurements: {len(re_values)}")
print("\nFirst 5 Re values (Ohms):", re_values[:5])
print("First 5 Rct values (Ohms):", rct_values[:5])
print("\nLast 5 Re values (Ohms):", re_values[-5:])
print("Last 5 Rct values (Ohms):", rct_values[-5:])