"""Module for regret minimizers operating over sequence-form polytopes."""
from dataclasses import dataclass, field, KW_ONLY
from typing import Any

from abc import ABC

from noregret.regret_minimizers.probability_simplices import (
    DiscountedRegretMatching,
    RegretMatching,
    RegretMatchingPlus,
)
from noregret.regret_minimizers.regret_minimizers import (
    DiscountedRegretMinimizer,
    RegretMinimizer,
)


@dataclass
class SequenceFormPolytopeRegretMinimizer(RegretMinimizer, ABC):
    """Abstract base class for regret minimizers operating over
    sequence-form polytopes.
    """
    sequence_form_polytope: Any
    """Sequence-form polytope."""
    _: KW_ONLY
    previous_behavioral_strategy: Any = 0.0
    """Previous behavioral strategy."""
    previous_counterfactual_regrets: Any = 0.0
    """Previous counterfactual regrets."""
    cumulative_counterfactual_regrets: Any = 0.0
    """Cumulative counterfactual regrets."""
    behavioral_strategies: list[Any] = field(default_factory=list)
    """Behavioral strategies."""
    _next_behavioral_strategy: Any = None

    @property
    def dimension(self):
        """Return the dimension.

        :return: The dimension.
        """
        return self.sequence_form_polytope.column_count

    @property
    def next_behavioral_strategy(self):
        """Return the next behavioral strategy.

        :return: The next behavioral strategy.
        """
        return self._next_behavioral_strategy

    @next_behavioral_strategy.setter
    def next_behavioral_strategy(self, value):
        if self._next_behavioral_strategy is not None and value is not None:
            raise ValueError('next behavioral strategy already outputted')

        self._next_behavioral_strategy = value

    def observe(self, utility):
        super().observe(utility)

        b = self.next_behavioral_strategy
        self.next_behavioral_strategy = None
        self.previous_behavioral_strategy = b
        r = self.sequence_form_polytope.counterfactual_regrets(b, utility)
        self.previous_counterfactual_regrets = r
        self.cumulative_counterfactual_regrets += r

        if not self.is_time_symmetric:
            self.behavioral_strategies.append(b)


@dataclass
class CounterfactualRegretMinimization(SequenceFormPolytopeRegretMinimizer):
    """Class for counterfactual regret minimization (CFR)."""
    regret_minimizer_type: Any = RegretMatching
    """Regret minimizer type."""

    def _theta(self, m):
        if m is False:
            theta = self.cumulative_counterfactual_regrets
        else:
            if m is True:
                m = self.previous_utility

            r = self.sequence_form_polytope.counterfactual_regrets(
                self.previous_behavioral_strategy,
                m,
            )
            theta = r + self.cumulative_counterfactual_regrets

        if self.kernel.numpy.isscalar(theta):
            theta = self.kernel.numpy.full(self.dimension - 1, theta)

        return theta.clip(0)

    def output(self, prediction=False):
        theta = self._theta(prediction)
        normalize = self.sequence_form_polytope.normalize
        self.next_behavioral_strategy = normalize(theta)
        self.next_strategy = self.sequence_form_polytope.to_sequence_form(
            self.next_behavioral_strategy,
        )

        return self.next_strategy


@dataclass
class CounterfactualRegretMinimizationPlus(CounterfactualRegretMinimization):
    """Class for counterfactual regret minimization+ (CFR+)."""
    regret_minimizer_type: Any = RegretMatchingPlus
    _: KW_ONLY
    floored_cumulative_counterfactual_regrets: Any = 0.0
    """Floored cumulative counterfactual regrets."""
    gamma: int = 1

    def _theta(self, m):
        if m is False:
            theta = self.floored_cumulative_counterfactual_regrets
        else:
            if m is True:
                m = self.previous_utility

            r = self.sequence_form_polytope.counterfactual_regrets(
                self.previous_behavioral_strategy,
                m,
            )
            theta = r + self.floored_cumulative_counterfactual_regrets
            theta = theta.clip(0)

        if self.kernel.numpy.isscalar(theta):
            theta = self.kernel.numpy.full(self.dimension - 1, theta)

        return theta

    def observe(self, utility):
        super().observe(utility)

        self.floored_cumulative_counterfactual_regrets += (
            self.previous_counterfactual_regrets
        )
        r_plus = self.floored_cumulative_counterfactual_regrets

        r_plus.clip(0, out=r_plus)


@dataclass
class DiscountedCounterfactualRegretMinimization(
        CounterfactualRegretMinimization,
        DiscountedRegretMinimizer,
):
    """Class for discounted counterfactual regret minimization+ (DCFR)."""
    regret_minimizer_type: Any = DiscountedRegretMatching
    _: KW_ONLY
    discounted_counterfactual_regrets: Any = 0.0
    """Discounted counterfactual regrets."""

    def _theta(self, m):
        if m is False:
            theta = self.discounted_counterfactual_regrets
        else:
            if m is True:
                m = self.previous_utility

            r = self.sequence_form_polytope.counterfactual_regrets(
                self.previous_behavioral_strategy,
                m,
            )
            theta = r + self.discounted_counterfactual_regrets
            T = self.iteration_count + 1
            theta[theta > 0] *= T ** self.alpha / (T ** self.alpha + 1)
            theta[theta < 0] *= T ** self.beta / (T ** self.beta + 1)

        if self.kernel.numpy.isscalar(theta):
            theta = self.kernel.numpy.full(self.dimension - 1, theta)

        return theta.clip(0)

    def observe(self, utility):
        super().observe(utility)

        self.discounted_counterfactual_regrets += (
            self.previous_counterfactual_regrets
        )
        r = self.discounted_counterfactual_regrets
        T = self.iteration_count
        r[r > 0] *= T ** self.alpha / (T ** self.alpha + 1)
        r[r < 0] *= T ** self.beta / (T ** self.beta + 1)
