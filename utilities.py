from dataclasses import dataclass, field, KW_ONLY
from functools import partial, update_wrapper
from time import time
from typing import Any, ClassVar

from scipy.stats import sem
import noregret as nr
import numpy as np


@dataclass
class MethodTimer:
    instances: ClassVar[list[Any]] = []
    method: Any
    times: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.instances.append(self)
        update_wrapper(self, self.method)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self

        return partial(self.__call__, instance)

    def __call__(self, *args, **kwargs):
        initial_time = time()
        output = self.method(*args, **kwargs)
        time_ = time() - initial_time

        self.times.append(time_)

        return output

    @property
    def total(self):
        return sum(self.times)

    @property
    def average(self):
        return np.mean(self.times)

    @property
    def standard_error(self):
        return sem(self.times)


@dataclass
class TimedCFR:
    kernel: nr.Kernel
    sequence_form_polytope: nr.SequenceFormPolytope
    _: KW_ONLY
    iteration_count: int = 0
    average_strategy: Any = 0.0
    next_strategy: Any = field(default=None, init=False)
    next_behavioral_strategy: Any = None
    cumulative_counterfactual_regrets: Any = 0.0

    @property
    def dimension(self):
        return self.sequence_form_polytope.column_count

    @MethodTimer
    def calculate_behavioral_strategy(self):
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        theta = self.cumulative_counterfactual_regrets

        if np.isscalar(theta):
            theta = np.full(self.dimension - 1, theta, dtype)

        theta = theta.clip(0)
        self.next_behavioral_strategy = self.sequence_form_polytope.normalize(
            theta,
        )

        return self.next_behavioral_strategy

    @MethodTimer
    def convert_to_sequence_form(self):
        self.next_strategy = self.sequence_form_polytope.to_sequence_form(
            self.next_behavioral_strategy,
        )

    def output(self, prediction=False):
        assert prediction is False

        self.calculate_behavioral_strategy()
        self.convert_to_sequence_form()

        return self.next_strategy

    @MethodTimer
    def update_average_strategy(self):
        x = self.next_strategy
        x_bar = self.average_strategy
        self.iteration_count += 1
        self.next_strategy = None
        self.average_strategy += (x - x_bar) / self.iteration_count

    @MethodTimer
    def memoize_counterfactual_utility(self, utility):
        b = self.next_behavioral_strategy
        self.next_behavioral_strategy = None
        sfp = self.sequence_form_polytope
        A = sfp._A.copy()
        A[sfp._R, sfp._C] = b

        for L_R, L_C_B, L_B in zip(
                sfp._L_R[::-1],
                sfp._L_C_B2[::-1],
                sfp._L_B[::-1],
        ):
            utility[L_C_B] += A[L_R] @ utility @ L_B

        return A, utility

    @MethodTimer
    def update_counterfactual_regrets(self, A, utility):
        utility -= A @ utility @ self.sequence_form_polytope._A
        r = utility[1:]
        self.cumulative_counterfactual_regrets += r

        return r

    def observe(self, utility):
        self.update_average_strategy()

        A, utility = self.memoize_counterfactual_utility(utility)

        self.update_counterfactual_regrets(A, utility)
