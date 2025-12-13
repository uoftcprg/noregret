from abc import ABC, abstractmethod
from dataclasses import dataclass, field, KW_ONLY
from functools import partial
from itertools import repeat
from math import inf, isinf
from typing import Any

from scipy.special import softmax
import numpy as np

from noregret.utilities import (
    euclidean_projection_on_probability_simplex,
    split,
    stationary_distribution,
)


@dataclass
class RegretMinimizer(ABC):
    """Regret minimizer."""

    dimension: Any
    _: KW_ONLY
    gamma: Any = 0
    is_time_symmetric: bool = False
    iteration_count: Any = 0
    strategies: Any = field(default_factory=list)
    previous_strategy: Any = None
    weight_sum: Any = 0
    average_strategy: Any = None
    utilities: Any = field(default_factory=list)
    previous_utility: Any = None
    cumulative_utility: Any = None

    def __post_init__(self):
        if self.previous_strategy is None:
            self.previous_strategy = np.full(
                self.dimension,
                1 / self.dimension,
            )

        if self.average_strategy is None:
            self.average_strategy = np.full(self.dimension, 1 / self.dimension)

        if self.previous_utility is None:
            self.previous_utility = np.zeros(self.dimension)

        if self.cumulative_utility is None:
            self.cumulative_utility = np.zeros(self.dimension)

    @abstractmethod
    def next_strategy(self, prediction=False):
        pass

    def observe_utility(self, utility):
        if len(self.strategies) == len(self.utilities):
            raise ValueError('next strategy not yet outputted')
        elif len(self.strategies) != len(self.utilities) + 1:
            raise ValueError('more than one strategies were outputted')

        strategy = self.strategies[-1]
        self.iteration_count += 1

        if self.is_time_symmetric:
            self.strategies.clear()

        self.previous_strategy = strategy.copy()

        self.update_average_strategy()

        if not self.is_time_symmetric:
            self.utilities.append(utility.copy())

        self.previous_utility = utility.copy()
        self.cumulative_utility += utility

    def update_average_strategy(self):
        if self.gamma > 0 and isinf(self.gamma):
            weight = inf
        else:
            weight = self.iteration_count ** self.gamma

        self.weight_sum += weight

        if isinf(weight):
            self.average_strategy = self.previous_strategy
        else:
            self.average_strategy += (
                weight
                * (self.previous_strategy - self.average_strategy)
                / self.weight_sum
            )


@dataclass
class ProbabilitySimplexRegretMinimizer(RegretMinimizer, ABC):
    """Regret minimizer.

    Assumes a linear utility function, and optimizes over the
    probability simplex.
    """

    _: KW_ONLY
    regrets: Any = field(default_factory=list)
    previous_regrets: Any = None
    cumulative_regrets: Any = None
    cumulative_regret: Any = 0

    def __post_init__(self):
        super().__post_init__()

        if self.previous_regrets is None:
            self.previous_regrets = np.zeros(self.dimension)

        if self.cumulative_regrets is None:
            self.cumulative_regrets = np.zeros(self.dimension)

    def observe_utility(self, utility):
        super().observe_utility(utility)

        regrets = utility - (self.previous_strategy @ utility)

        if not self.is_time_symmetric:
            self.regrets.append(regrets.copy())

        self.previous_regrets = regrets.copy()
        self.cumulative_regrets += regrets
        self.cumulative_regret = self.cumulative_regrets.max()


@dataclass
class FollowTheRegularizedLeader(ProbabilitySimplexRegretMinimizer, ABC):
    """Follow the regularized leader (FTLR)."""

    learning_rate: Any


@dataclass
class MultiplicativeWeightsUpdate(FollowTheRegularizedLeader):
    """Multiplicative weights update (MWU)."""

    def next_strategy(self, prediction=False):
        if prediction is False:
            theta = self.cumulative_utility
        else:
            if prediction is True:
                prediction = self.previous_utility

            theta = prediction + self.cumulative_utility

        strategy = softmax(self.learning_rate * theta)

        self.strategies.append(strategy)

        return strategy


@dataclass
class EuclideanRegularization(FollowTheRegularizedLeader):
    """Euclidean regularization."""

    def next_strategy(self, prediction=False):
        if prediction is False:
            theta = self.cumulative_utility
        else:
            if prediction is True:
                prediction = self.previous_utility

            theta = prediction + self.cumulative_utility

        strategy = euclidean_projection_on_probability_simplex(
            self.learning_rate * theta,
        )

        self.strategies.append(strategy)

        return strategy


@dataclass
class MirrorDescent(ProbabilitySimplexRegretMinimizer, ABC):
    """(Online) mirror descent (MR)."""

    learning_rate: Any


@dataclass
class OnlineGradientDescent(MirrorDescent):
    """Online gradient descent."""

    def next_strategy(self, prediction=False):
        if prediction is False or prediction is True:
            theta = self.previous_utility
        else:
            theta = prediction

        strategy = euclidean_projection_on_probability_simplex(
            self.previous_strategy + self.learning_rate * theta,
        )

        self.strategies.append(strategy)

        return strategy


@dataclass
class RegretMatching(ProbabilitySimplexRegretMinimizer):
    """Regret matching."""

    def next_strategy(self, prediction=False):
        if prediction is False:
            theta = self.cumulative_regrets
        else:
            if prediction is True:
                prediction = self.previous_utility

            theta = (
                prediction
                - prediction @ self.previous_strategy
                + self.cumulative_regrets
            )

        return self._next_strategy(theta.clip(0))

    def _next_strategy(self, unnormalized_strategy):
        if np.allclose(unnormalized_strategy, 0):
            strategy = np.full(self.dimension, 1 / self.dimension)
        else:
            strategy = unnormalized_strategy / unnormalized_strategy.sum()

        self.strategies.append(strategy)

        return strategy


@dataclass
class RegretMatchingPlus(RegretMatching):
    """Regret matching plus."""

    _: KW_ONLY
    cumulative_regrets_plus: Any = None

    def __post_init__(self):
        super().__post_init__()

        if self.cumulative_regrets_plus is None:
            self.cumulative_regrets_plus = np.zeros(self.dimension)

    def next_strategy(self, prediction=False):
        if prediction is False:
            theta = self.cumulative_regrets_plus
        else:
            if prediction is True:
                prediction = self.previous_utility

            theta = (
                prediction
                - prediction @ self.previous_strategy
                + self.cumulative_regrets_plus
            ).clip(0)

        return self._next_strategy(theta)

    def observe_utility(self, utility):
        super().observe_utility(utility)

        self.cumulative_regrets_plus += self.previous_regrets
        self.cumulative_regrets_plus = self.cumulative_regrets_plus.clip(0)


class DiscountedStrategyAveragingMixin:
    gamma: Any
    iteration_count: Any
    previous_strategy: Any
    weight_sum: Any
    average_strategy: Any

    def update_average_strategy(self):
        T = self.iteration_count
        weight = (T / (T + 1)) ** self.gamma
        self.weight_sum += weight
        self.average_strategy += (
            weight
            * (self.previous_strategy - self.average_strategy)
            / self.weight_sum
        )


@dataclass
class DiscountedRegretMatching(
        DiscountedStrategyAveragingMixin,
        RegretMatching,
):
    """Discounted regret matching (used in DCFR)."""

    _: KW_ONLY
    alpha: Any = 0
    beta: Any = 0
    discounted_regrets: Any = None

    def __post_init__(self):
        super().__post_init__()

        if self.discounted_regrets is None:
            self.discounted_regrets = np.zeros(self.dimension)

    def next_strategy(self, prediction=False):
        if prediction is False:
            theta = self.discounted_regrets
        else:
            if prediction is True:
                prediction = self.previous_utility

            theta = (
                prediction
                - prediction @ self.previous_strategy
                + self.discounted_regrets
            )
            T = self.iteration_count + 1
            theta[theta > 0] *= T ** self.alpha / (T ** self.alpha + 1)
            theta[theta < 0] *= T ** self.beta / (T ** self.beta + 1)

        return self._next_strategy(theta.clip(0))

    def observe_utility(self, utility):
        super().observe_utility(utility)

        self.discounted_regrets += self.previous_regrets
        T = self.iteration_count
        self.discounted_regrets[self.discounted_regrets > 0] *= (
            T ** self.alpha / (T ** self.alpha + 1)
        )
        self.discounted_regrets[self.discounted_regrets < 0] *= (
            T ** self.beta / (T ** self.beta + 1)
        )


@dataclass
class SwapRegretMinimizer(RegretMinimizer, ABC):
    """Swap regret minimizer."""


@dataclass
class ProbabilitySimplexSwapRegretMinimizer(SwapRegretMinimizer, ABC):
    """Swap regret minimizer.

    Optimizes over the probability simplex.
    """


@dataclass
class BlumMansour(ProbabilitySimplexSwapRegretMinimizer):
    """Blum-Mansour algorithm.

    External regret minimizers optimize over the probability simplex.
    """

    regret_minimizer_factory: Any = None
    _: KW_ONLY
    outputs: Any = field(init=False)
    external_regret_minimizers: Any = None

    def __post_init__(self):
        super().__post_init__()

        d = self.dimension
        self.outputs = np.full((d, d), 1 / d)

        if self.external_regret_minimizers is None:
            if self.regret_minimizer_factory is None:
                raise ValueError('regret minimizer factory not set')

            self.external_regret_minimizers = tuple(
                map(self.regret_minimizer_factory, repeat(d, d)),
            )

    def next_strategy(self, prediction=False):
        if prediction is False or prediction is True:
            for a, R in enumerate(self.external_regret_minimizers):
                self.outputs[:, a] = R.next_strategy(prediction)
        else:
            for a, R in enumerate(self.external_regret_minimizers):
                self.outputs[:, a] = R.next_strategy(
                    self.previous_strategy[a] * prediction,
                )

        strategy = stationary_distribution(self.outputs)

        self.strategies.append(strategy)

        return strategy

    def observe_utility(self, utility):
        super().observe_utility(utility)

        for a, R in enumerate(self.external_regret_minimizers):
            R.observe_utility(self.previous_strategy[a] * utility)


@dataclass
class SequenceFormPolytopeRegretMinimizer(RegretMinimizer, ABC):
    """Regret minimizer.

    Optimizes over the sequence-form polytope.
    """


@dataclass
class CounterfactualRegretMinimization(SequenceFormPolytopeRegretMinimizer):
    """Counterfactual regret minimization (CFR).

    Local regret minimizers optimize over the probability simplex.
    """

    tree_form_sequential_decision_process: Any
    regret_minimizer_factory: Any = RegretMatching
    _: KW_ONLY
    dimension: Any = field(init=False)
    behavioral_strategy: Any = field(init=False)
    previous_behavioral_strategy: Any = None
    local_regret_minimizers: Any = None

    def __post_init__(self):
        tfsdp = self.tree_form_sequential_decision_process
        self.dimension = len(tfsdp.sequences)

        super().__post_init__()

        self.behavioral_strategy = tfsdp.behavioral_uniform_strategy()

        if self.previous_behavioral_strategy is None:
            self.previous_behavioral_strategy = self.behavioral_strategy.copy()

        if self.local_regret_minimizers is None:
            if self.regret_minimizer_factory is None:
                raise ValueError('regret minimizer factory not set')

            self.local_regret_minimizers = {
                j: self.regret_minimizer_factory(len(tfsdp.actions[j]))
                for j in tfsdp.decision_points
            }

    def next_strategy(self, prediction=False):
        self.behavioral_strategy = {}

        if prediction is False or prediction is True:
            for j, R in self.local_regret_minimizers.items():
                self.behavioral_strategy[j] = R.next_strategy(prediction)
        else:
            counterfactual_predictions = (
                self
                .tree_form_sequential_decision_process
                .counterfactual_utilities(
                    self.previous_behavioral_strategy,
                    prediction,
                )
            )

            for j, R in self.local_regret_minimizers.items():
                self.behavioral_strategy[j] = R.next_strategy(
                    counterfactual_predictions[j],
                )

        strategy = (
            self
            .tree_form_sequential_decision_process
            .behavioral_to_sequence_form(self.behavioral_strategy)
        )

        self.strategies.append(strategy)

        return strategy

    def observe_utility(self, utility):
        super().observe_utility(utility)

        counterfactual_utilities = (
            self
            .tree_form_sequential_decision_process
            .counterfactual_utilities(self.behavioral_strategy, utility)
        )

        for j, R in self.local_regret_minimizers.items():
            R.observe_utility(counterfactual_utilities[j])

        self.previous_behavioral_strategy = self.behavioral_strategy.copy()


@dataclass
class CounterfactualRegretMinimizationPlus(CounterfactualRegretMinimization):
    """Counterfactual regret minimization plus (CFR+)."""

    regret_minimizer_factory: Any = RegretMatchingPlus
    _: KW_ONLY
    gamma: Any = 1


@dataclass
class DiscountedCounterfactualRegretMinimization(
        DiscountedStrategyAveragingMixin,
        CounterfactualRegretMinimization,
):
    """Discounted counterfactual regret minimization (DCFR)."""

    regret_minimizer_factory: Any = None
    _: KW_ONLY
    alpha: Any = 1.5
    beta: Any = 0
    gamma: Any = 2

    def __post_init__(self):
        if self.regret_minimizer_factory is None:
            self.regret_minimizer_factory = partial(
                DiscountedRegretMatching,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

        super().__post_init__()


@dataclass
class RegretCircuit(RegretMinimizer):
    """Regret circuit.

    Assumes a linear utility function.
    """


@dataclass
class CartesianProductRegretCircuit(RegretCircuit):
    """Regret circuit for the cartesian product operation.

    Optimizes over the cartesian product of sets, over which the components
    optimize.
    """

    component_regret_minimizers: Any
    _: KW_ONLY
    dimension: Any = field(init=False)

    def __post_init__(self):
        self.dimension = sum(self.dimensions)

        super().__post_init__()

    @property
    def dimensions(self):
        return np.array(
            [R.dimension for R in self.component_regret_minimizers],
        )

    def next_strategy(self, prediction=False):
        if prediction is False or prediction is True:
            predictions = repeat(prediction)
        else:
            predictions = split(prediction, self.dimensions)

        strategy = []

        for R, m in zip(self.component_regret_minimizers, predictions):
            strategy.append(R.next_strategy(m))

        strategy = np.concatenate(strategy)

        self.strategies.append(strategy)

        return strategy

    def observe_utility(self, utility):
        super().observe_utility(utility)

        for R, utility in zip(
                self.component_regret_minimizers,
                split(utility, self.dimensions),
        ):
            R.observe_utility(utility)


@dataclass
class ConvexHullRegretCircuit(RegretCircuit):
    """Regret circuit for the convex hull operation.

    Optimizes over the convex hull of sets, over which the components
    optimize.
    """

    component_regret_minimizers: Any
    regret_minimizer_factory: Any = None
    _: KW_ONLY
    dimension: Any = field(init=False)
    outputs: Any = field(init=False)
    previous_outputs: Any = None
    mixing_regret_minimizer: Any = None

    def __post_init__(self):
        if self.component_regret_minimizers:
            self.dimension = self.component_regret_minimizers[0].dimension
        else:
            self.dimension = 0

        super().__post_init__()

        self.outputs = np.full(
            (len(self.component_regret_minimizers), self.dimension),
            1 / self.dimension,
        )

        if self.previous_outputs is None:
            self.previous_outputs = self.outputs.copy()

        if self.mixing_regret_minimizer is None:
            if self.regret_minimizer_factory is None:
                raise ValueError('regret minimizer factory not set')

            self.mixing_regret_minimizer = self.regret_minimizer_factory(
                len(self.component_regret_minimizers),
            )

    def next_strategy(self, prediction=False):
        for i, R in enumerate(self.component_regret_minimizers):
            self.outputs[i] = R.next_strategy(prediction)

        if prediction is False or prediction is True:
            m = prediction
        else:
            m = self.previous_outputs @ prediction

        probabilities = self.mixing_regret_minimizer.next_strategy(m)
        strategy = self.outputs.T @ probabilities

        self.strategies.append(strategy)

        return strategy

    def observe_utility(self, utility):
        super().observe_utility(utility)

        for R in self.component_regret_minimizers:
            R.observe_utility(utility)

        self.previous_outputs = self.outputs.copy()

        self.mixing_regret_minimizer.observe_utility(self.outputs @ utility)
