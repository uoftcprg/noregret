"""Module for regret minimizers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, KW_ONLY
from math import inf, isinf
from typing import Any, Literal

from noregret.kernels import Kernel


@dataclass
class RegretMinimizer(ABC):
    """Abstract base class for regret minimizers.

    We use the game-theoretic language of `strategy' and `utility',
    instead of the more standard decision-theoretic language of
    `decision' and `loss'.
    """
    kernel: Kernel
    """Kernel."""
    _: KW_ONLY
    gamma: int | Literal[inf] = 0
    """Gamma."""
    is_time_symmetric: bool = True
    """Whether the regret minimizer is time symmetric."""
    iteration_count: int = 0
    """Number of iterations."""
    previous_strategy: Any = 0.0
    """Previous strategy."""
    weight_sum: int | Literal[inf] = 0
    """Weight sum."""
    average_strategy: Any = 0.0
    """Average strategy."""
    previous_utility: Any = 0.0
    """Previous utility."""
    cumulative_utility: Any = 0.0
    """Cumulative utility."""
    strategies: list[Any] = field(default_factory=list)
    """Strategies."""
    utilities: list[Any] = field(default_factory=list)
    """Utilities."""
    _next_strategy: Any = None

    @property
    def next_strategy(self):
        """Return the next strategy.

        :return: The next strategy.
        """
        return self._next_strategy

    @next_strategy.setter
    def next_strategy(self, value):
        if self._next_strategy is not None and value is not None:
            raise ValueError('next strategy already outputted')

        self._next_strategy = value

    @abstractmethod
    def output(self, prediction=False):
        """Output the next strategy.

        A prediction can optionally be given to facilitate optimism.

        :param prediction: Optional prediction; defaults to ``False''
                           for no optimism. If ``True'', the previous
                           utility is used.
        :return: Next strategy.
        """

    def observe(self, utility):
        """Observe the utility.

        :param utility: Observed utility.
        :return: ``None``.
        """
        if self.next_strategy is None:
            raise ValueError('next strategy not yet outputted')

        self.iteration_count += 1
        x = self.next_strategy.copy()
        self.next_strategy = None
        self.previous_strategy = x

        self._update_average_strategy()

        self.previous_utility = utility
        self.cumulative_utility += utility

        if not self.is_time_symmetric:
            self.strategies.append(x)
            self.utilities.append(utility)

    def _update_average_strategy(self):
        if self.gamma > 0 and isinf(self.gamma):
            self.weight_sum = inf
            self.average_strategy = self.previous_strategy
        else:
            w = self.iteration_count ** self.gamma
            self.weight_sum += w
            x = self.previous_strategy
            x_bar = self.average_strategy
            self.average_strategy += w * (x - x_bar) / self.weight_sum


class DiscountedRegretMinimizer(RegretMinimizer, ABC):
    """Abstract base class for discounted regret minimizers."""
    _: KW_ONLY
    alpha: Any = 1.5
    """Alpha."""
    beta: Any = 0.0
    """Beta."""
    gamma: int = 2

    def _update_average_strategy(self):
        if self.gamma > 0 and isinf(self.gamma):
            self.weight_sum = inf
            self.average_strategy = self.previous_strategy
        else:
            T = self.iteration_count
            w = (T / (T + 1)) ** self.gamma
            self.weight_sum += w
            x = self.previous_strategy
            x_bar = self.average_strategy
            self.average_strategy += w * (x - x_bar) / self.weight_sum


@dataclass
class SwapRegretMinimizer(RegretMinimizer, ABC):
    """Abstract base class for swap regret minimizers."""
