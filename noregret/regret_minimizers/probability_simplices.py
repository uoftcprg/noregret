"""Module for regret minimizers operating over probability simplices."""
from abc import ABC
from dataclasses import dataclass, field, KW_ONLY
from functools import partial
from itertools import repeat
from typing import Any

from noregret.regret_minimizers.regret_minimizers import (
    DiscountedRegretMinimizer,
    RegretMinimizer,
    SwapRegretMinimizer,
)


@dataclass
class ProbabilitySimplexRegretMinimizer(RegretMinimizer, ABC):
    """Abstract base class for regret minimizers operating over
    probability simplices.
    """
    dimension: int
    """Dimension."""
    _: KW_ONLY
    previous_regrets: Any = 0.0
    """Previous regrets."""
    cumulative_regrets: Any = 0.0
    """Cumulative regrets."""

    def regrets(self, index):
        """Return (instantaneous) regrets at a given index.

        :param index: index (0-indexed).
        :return: Regrets at the index.
        """
        if self.is_time_symmetric:
            raise ValueError('time symmetric')

        x = self.strategies[index]
        u = self.utilities[index]

        return u - x @ u

    @property
    def cumulative_regret(self):
        """Return the cumulative regret.

        :return: Cumulative regret.
        """
        return self.cumulative_regrets.max()

    def observe(self, utility):
        super().observe(utility)

        r = utility - (self.previous_strategy @ utility)
        self.previous_regrets = r
        self.cumulative_regrets += r


@dataclass
class FollowTheRegularizedLeader(ProbabilitySimplexRegretMinimizer, ABC):
    """Abstract base class for follow the regularized leader (FTLR)."""
    learning_rate: float
    """Learning rate."""

    def _theta(self, m):
        if m is False:
            theta = self.cumulative_utility
        else:
            if m is True:
                m = self.previous_utility

            theta = m + self.cumulative_utility

        if self.kernel.numpy.isscalar(theta):
            theta = self.kernel.numpy.full(self.dimension, theta)

        return theta


@dataclass
class MultiplicativeWeightsUpdate(FollowTheRegularizedLeader):
    """Class for multiplicative weights update (MWU)."""

    def output(self, prediction=False):
        theta = self._theta(prediction)
        sigma = self.kernel.scipy.special.softmax
        self.next_strategy = sigma(self.learning_rate * theta)

        return self.next_strategy


@dataclass
class EuclideanRegularization(FollowTheRegularizedLeader):
    """Class for Euclidean regularization (ER)."""

    def output(self, prediction=False):
        theta = self._theta(prediction)
        pi = self.kernel.euclidean_projection_on_probability_simplex
        self.next_strategy = pi(self.learning_rate * theta)

        return self.next_strategy


@dataclass
class MirrorDescent(ProbabilitySimplexRegretMinimizer, ABC):
    """Class for (Online) mirror descent (MR)."""
    learning_rate: float
    """Learning rate."""


@dataclass
class OnlineGradientDescent(MirrorDescent):
    """Online gradient descent (OGD)."""

    def output(self, prediction=False):
        if prediction is False or prediction is True:
            theta = self.previous_utility
        else:
            theta = prediction

        if self.kernel.numpy.isscalar(theta):
            theta = self.kernel.numpy.full(self.dimension, theta)

        pi = self.kernel.euclidean_projection_on_probability_simplex
        lr = self.learning_rate
        self.next_strategy = pi(self.previous_strategy + lr * theta)

        return self.next_strategy


@dataclass
class RegretMatching(ProbabilitySimplexRegretMinimizer):
    """Class for regret matching (RM)."""

    def _theta(self, m):
        if m is False:
            theta = self.cumulative_regrets
        else:
            if m is True:
                m = self.previous_utility

            r = m - self.kernel.numpy.dot(m, self.previous_strategy)
            theta = r + self.cumulative_regrets

        if self.kernel.numpy.isscalar(theta):
            theta = self.kernel.numpy.full(self.dimension, theta)

        return theta.clip(0)

    def output(self, prediction=False):
        theta = self._theta(prediction)
        np = self.kernel.numpy

        if np.allclose(theta, 0):
            self.next_strategy = np.full(self.dimension, 1 / self.dimension)
        else:
            self.next_strategy = theta / theta.sum()

        return self.next_strategy


@dataclass
class RegretMatchingPlus(RegretMatching):
    """Class for regret matching+ (RM+)."""
    _: KW_ONLY
    floored_cumulative_regrets: Any = 0.0
    """Floored cumulative regrets."""

    def _theta(self, m):
        if m is False:
            theta = self.floored_cumulative_regrets
        else:
            if m is True:
                m = self.previous_utility

            if self.kernel.numpy.isscalar(m):
                m = self.kernel.numpy.full(self.dimension, m)

            r = m - self.kernel.numpy.dot(m, self.previous_strategy)
            theta = r + self.floored_cumulative_regrets
            theta = theta.clip(0)

        if self.kernel.numpy.isscalar(theta):
            theta = self.kernel.numpy.full(self.dimension, theta)

        return theta

    def observe(self, utility):
        super().observe(utility)

        self.floored_cumulative_regrets += self.previous_regrets
        r_plus = self.floored_cumulative_regrets

        r_plus.clip(0, out=r_plus)


@dataclass
class DiscountedRegretMatching(RegretMatching, DiscountedRegretMinimizer):
    """Class for discounted regret matching (DRM)."""
    _: KW_ONLY
    discounted_regrets: Any = 0.0
    """Discounted regrets."""

    def _theta(self, m):
        if m is False:
            theta = self.discounted_regrets
        else:
            if m is True:
                m = self.previous_utility

            if self.kernel.numpy.isscalar(m):
                m = self.kernel.numpy.full(self.dimension, m)

            r = m - self.kernel.numpy.dot(m, self.previous_strategy)
            theta = r + self.discounted_regrets
            T = self.iteration_count + 1
            theta[theta > 0] *= T ** self.alpha / (T ** self.alpha + 1)
            theta[theta < 0] *= T ** self.beta / (T ** self.beta + 1)

        if self.kernel.numpy.isscalar(theta):
            theta = self.kernel.numpy.full(self.dimension, theta)

        return theta.clip(0)

    def observe(self, utility):
        super().observe(utility)

        self.discounted_regrets += self.previous_regrets
        r = self.discounted_regrets
        T = self.iteration_count
        r[r > 0] *= T ** self.alpha / (T ** self.alpha + 1)
        r[r < 0] *= T ** self.beta / (T ** self.beta + 1)


@dataclass
class ProbabilitySimplexSwapRegretMinimizer(
        ProbabilitySimplexRegretMinimizer,
        SwapRegretMinimizer,
        ABC,
):
    """Abstract base class for swap regret minimizers operating over
    probability simplices.
    """


@dataclass
class BlumMansour(ProbabilitySimplexSwapRegretMinimizer):
    """Class for the Blum-Mansour (BM) algorithm."""
    regret_minimizer_type: type[ProbabilitySimplexRegretMinimizer] = None
    """Regret minimizer type."""
    _: KW_ONLY
    external_regret_minimizers: Any = field(init=False)
    """External regret minimizeres."""

    def __post_init__(self):
        n = self.dimension
        R_type = partial(self.regret_minimizer_type, self.kernel)
        self.external_regret_minimizers = tuple(map(R_type, repeat(n, n)))

    def output(self, prediction=False):
        if prediction is not False:
            raise NotImplementedError

        np = self.kernel.numpy
        M = np.full((self.dimension, self.dimension), 1 / self.dimension)

        for a, R in enumerate(self.external_regret_minimizers):
            M[:, a] = R.output()

        self.next_strategy = self.kernel.stationary_distribution(M.T)

        return self.next_strategy

    def observe(self, utility):
        super().observe(utility)

        for a, R in enumerate(self.external_regret_minimizers):
            R.observe(self.previous_strategy[a] * utility)
