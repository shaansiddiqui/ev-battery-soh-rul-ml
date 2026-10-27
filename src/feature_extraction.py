import numpy as np

def extract_features(cycle_entry):
    data = cycle_entry["data"][0, 0]

    voltage = data["Voltage_measured"][0]
    current = data["Current_measured"][0]
    temperature = data["Temperature_measured"][0]
    time = data["Time"][0]

    features = {
        "mean_voltage": np.mean(voltage),
        "min_voltage": np.min(voltage),
        "mean_current": np.mean(current),
        "mean_temperature": np.mean(temperature),
        "max_temperature": np.max(temperature),
        "discharge_duration": time[-1] - time[0]
    }

    return features