from LiteEFG.baselines.baseline import _baseline
from scipy.stats import sem
import LiteEFG
import numpy as np


def iteration_times(times):
    times[1:] = np.diff(times)

    return times


def iteration_time(times):
    times = iteration_times(times)

    return np.mean(times), sem(times)


def delogarithmize(values, count):
    values2 = [None] * count

    for i, value in enumerate(values):
        index = min(2 ** i, count) - 1
        values2[index] = value

    return values2


class graph(_baseline):
    def __init__(self):
        super().__init__()

        with LiteEFG.backward(True):
            expectation = LiteEFG.const(1, 0.0)
            self.strategy = LiteEFG.const(
                self.action_set_size,
                1.0 / self.action_set_size,
            )
            self.regret_buffer = LiteEFG.const(self.action_set_size, 0.0)

        with LiteEFG.backward():
            counterfactual_value = (
                LiteEFG.aggregate(expectation, 'sum')
                + self.utility
            )

            expectation.inplace(
                LiteEFG.dot(counterfactual_value, self.strategy),
            )
            self.regret_buffer.inplace(
                self.regret_buffer + counterfactual_value - expectation,
            )
            self.strategy.inplace(
                LiteEFG.normalize(self.regret_buffer, 1.0, True),
            )

    def update_graph(self, env):
        env.update(self.strategy, 1)
        env.update(self.strategy, 2)

    def current_strategy(self):
        return self.strategy
