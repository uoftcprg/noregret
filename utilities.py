from scipy.stats import sem
import numpy as np


def iteration_times(times):
    times[1:] = np.diff(times)

    return times


def iteration_time(times):
    times = iteration_times(times)

    return np.mean(times), sem(times)
