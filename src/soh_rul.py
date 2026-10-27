def calculate_soh(capacities):
    initial_capacity = capacities[0]
    soh = [(c / initial_capacity) * 100 for c in capacities]
    return soh, initial_capacity


def find_eol_cycle(cycle_numbers, soh, eol_threshold=80.0):
    eol_cycle = None
    for cn, s in zip(cycle_numbers, soh):
        if s <= eol_threshold:
            eol_cycle = cn
            break
    return eol_cycle


def calculate_rul(cycle_numbers, eol_cycle):
    rul = [eol_cycle - cn if eol_cycle is not None else None for cn in cycle_numbers]
    return rul


def clip_rul(rul):
    return [max(0, r) for r in rul]