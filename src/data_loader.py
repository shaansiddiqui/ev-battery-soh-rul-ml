from scipy.io import loadmat

def load_battery_data(file_path, battery_key):
    data = loadmat(file_path)
    cycle_data = data[battery_key]["cycle"][0, 0][0]

    capacities = []
    cycle_numbers = []

    cycle_count = 0
    for i in range(cycle_data.shape[0]):
        if cycle_data[i]["type"][0] == "discharge":
            cycle_count += 1
            cap = cycle_data[i]["data"][0, 0]["Capacity"][0][0]
            capacities.append(cap)
            cycle_numbers.append(cycle_count)

    return capacities, cycle_numbers, cycle_data