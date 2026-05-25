"""Module for regret minimizers operating over sequence-form polytopes."""
from dataclasses import dataclass, field, KW_ONLY
from typing import Any

from abc import ABC

from noregret.regret_minimizers.probability_simplices import (
    ProbabilitySimplexRegretMinimizer,
    RegretMatching,
)
from noregret.regret_minimizers.regret_minimizers import (
    DiscountedRegretMinimizer,
    RegretMinimizer,
)
from noregret.sequence_form_polytopes import SequenceFormPolytope


@dataclass
class SequenceFormPolytopeRegretMinimizer(RegretMinimizer, ABC):
    """Abstract base class for regret minimizers operating over
    sequence-form polytopes.
    """
    sequence_form_polytope: SequenceFormPolytope
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

    def _theta(self, m):
        np = self.kernel.numpy
        dtype = self.kernel.data_type

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

        if np.isscalar(theta):
            theta = np.full(self.dimension - 1, theta, dtype)

        return theta.clip(0)

    def output(self, prediction=False):
        theta = self._theta(prediction)
        b = self.sequence_form_polytope.normalize(theta)
        self.next_behavioral_strategy = b
        self.next_strategy = self.sequence_form_polytope.to_sequence_form(b)

        return self.next_strategy


@dataclass
class CounterfactualRegretMinimizationPlus(CounterfactualRegretMinimization):
    """Class for counterfactual regret minimization+ (CFR+)."""
    _: KW_ONLY
    floored_cumulative_counterfactual_regrets: Any = 0.0
    """Floored cumulative counterfactual regrets."""
    gamma: int = 1

    def _theta(self, m):
        np = self.kernel.numpy
        dtype = self.kernel.data_type

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

        if np.isscalar(theta):
            theta = np.full(self.dimension - 1, theta, dtype)

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
    _: KW_ONLY
    discounted_cumulative_counterfactual_regrets: Any = 0.0
    """Discounted cumulative counterfactual regrets."""

    def _theta(self, m):
        np = self.kernel.numpy
        dtype = self.kernel.data_type

        if m is False:
            theta = self.discounted_cumulative_counterfactual_regrets
        else:
            if m is True:
                m = self.previous_utility

            r = self.sequence_form_polytope.counterfactual_regrets(
                self.previous_behavioral_strategy,
                m,
            )
            theta = r + self.discounted_cumulative_counterfactual_regrets
            T = self.iteration_count + 1
            theta[theta > 0] *= T ** self.alpha / (T ** self.alpha + 1)
            theta[theta < 0] *= T ** self.beta / (T ** self.beta + 1)

        if np.isscalar(theta):
            theta = np.full(self.dimension - 1, theta, dtype)

        return theta.clip(0)

    def observe(self, utility):
        super().observe(utility)

        self.discounted_cumulative_counterfactual_regrets += (
            self.previous_counterfactual_regrets
        )
        r = self.discounted_cumulative_counterfactual_regrets
        T = self.iteration_count
        r[r > 0] *= T ** self.alpha / (T ** self.alpha + 1)
        r[r < 0] *= T ** self.beta / (T ** self.beta + 1)


@dataclass
class CounterfactualRegretMinimization2(SequenceFormPolytopeRegretMinimizer):
    """Class for counterfactual regret minimization (CFR).

    This is an alternative to :class:`CounterfactualRegretMinimization`.

    Do **not** use this class unless it is absolutely necessary.

    Main advantage: Arbitrary local regret minimizers.

    Main disadvantage: **Slow** and unparallelizable.
    """
    regret_minimizer_type: type[ProbabilitySimplexRegretMinimizer] = (
        RegretMatching
    )
    """Regret minimizer type."""
    _: KW_ONLY
    regret_minimizers: dict[str, ProbabilitySimplexRegretMinimizer] = field(
        default_factory=dict,
        init=False,
    )
    """Regret minimizers."""

    def __post_init__(self):
        super().__post_init__()

        R_type = self.regret_minimizer_type
        A = self.sequence_form_polytope.actions
        J = self.sequence_form_polytope.decision_points

        for j in J:
            self.regret_minimizers[j] = R_type(self.kernel, len(A[j]))

    def output(self, prediction=False):
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        A = self.sequence_form_polytope.actions
        J = self.sequence_form_polytope.decision_points
        seqs = self.sequence_form_polytope.non_empty_sequences

        if prediction is False or prediction is True:
            predictions = {j: prediction for j in J}
        else:
            predictions = {}
            m = self.sequence_form_polytope.counterfactual_utilities(
                prediction,
            )

            for j in J:
                m_j = []

                for a in A[j]:
                    m_j.append(m[seqs.index((j, a))])

                predictions[j] = np.array(m_j, dtype)

        b = np.empty(len(seqs), dtype)

        for j, R in self.regret_minimizers.items():
            x = R.output(predictions[j])

            for a, p in zip(A[j], x):
                b[seqs.index((j, a))] = p

        self.next_behavioral_strategy = b
        self.next_strategy = self.sequence_form_polytope.to_sequence_form(b)

        return self.next_strategy

    def observe(self, utility):
        super().observe(utility)

        np = self.kernel.numpy
        dtype = self.kernel.data_type
        A = self.sequence_form_polytope.actions
        J = self.sequence_form_polytope.decision_points
        seqs = self.sequence_form_polytope.non_empty_sequences
        u = self.sequence_form_polytope.counterfactual_utilities(
            self.previous_behavioral_strategy,
            utility,
        )
        counterfactual_utilities = {}

        for j in J:
            u_j = []

            for a in A[j]:
                u_j.append(u[seqs.index((j, a))])

            counterfactual_utilities[j] = np.array(u_j, dtype)

        for j, R in self.regret_minimizers.items():
            R.observe(counterfactual_utilities[j])
