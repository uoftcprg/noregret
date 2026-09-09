import numpy as np


def iteration_times(times):
    times[1:] = np.diff(times)

    return times
