"""Module for regret minimizers operating over sequence-form polytopes."""
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from noregret.games.black_box import BlackBoxGame
from noregret.kernels import Kernel
from noregret.regret_minimizers.probability_simplices import (
    ProbabilitySimplexRegretMinimizer,
    RegretMatching,
)


@dataclass
class StochasticRegretMinimizer(ABC):
    """Abstract base class for stochastic regret minimizers."""
    kernel: Kernel
    """Kernel."""
    game: BlackBoxGame
    """Game."""
    regret_minimizer_type: type[ProbabilitySimplexRegretMinimizer]
    """Regret minimizer type."""
    reference_strategy_profile: Callable[[Any], Any] | None = None
    """Reference strategy profile."""
    sample_count: int = field(default=0, init=False)
    """number of samples."""
    next_sample_count: int = field(default=0, init=False)
    """Next number of samples."""
    node_visit_count: int = field(default=0, init=False)
    """Number of node visits."""
    next_node_visit_count: int = field(default=0, init=False)
    """Next number of node visits."""
    regret_minimizers: dict[str, ProbabilitySimplexRegretMinimizer] = field(
        default_factory=dict,
        init=False,
    )

    def regret_minimizer(self, node):
        """Return the regret minimizer given an information set.

        :param node: Node.
        :return: Regret minimizer.
        """
        j = self.game.information_set(node)

        if j not in self.regret_minimizers:
            self.regret_minimizers[j] = self.regret_minimizer_type(
                self.kernel,
                len(self.game.actions(node)),
            )

        return self.regret_minimizers[j]

    def average_action_probabilities(self, node):
        """Return the average action probabilities given a node.

        :param node: Node.
        :return: Average action probabilities.
        """
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        R = self.regret_minimizer(node)
        ps = R.average_strategy

        if np.isscalar(ps):
            ps = np.full(R.dimension, 1 / R.dimension, dtype)

        return ps

    def _action_probabilities(self, h):
        R = self.regret_minimizer(h)
        ps = R.next_strategy

        if ps is None:
            ps = R.output()

        return ps

    def _external_sampling(self, i, us, h):
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        self.next_node_visit_count += 1
        u = self.game.utility(h, i)
        A = self.game.actions(h)

        if A:
            i_prime = self.game.player(h)

            if i_prime is None:
                ps = self.game.chance_probabilities(h)
            else:
                ps = self._action_probabilities(h)

            if i_prime == i:
                u_primes = []

                for a in A:
                    h_prime = self.game.apply(h, a)

                    u_primes.append(self._external_sampling(i, us, h_prime))

                j = self.game.information_set(h)
                us[j] = np.array(u_primes, dtype)
                u += us[j] @ ps
            else:
                a = np.random.choice(A, p=ps).item()
                h_prime = self.game.apply(h, a)
                u += self._external_sampling(i, us, h_prime)

        return u

    def _external_sampling2(self, player):
        us = {}

        self._external_sampling(player, us, self.game.root_node)

        return us

    def _outcome_sampling(self, i, us, h, p):
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        self.next_node_visit_count += 1
        u = self.game.utility(h, i) / p
        A = self.game.actions(h)

        if A:
            i_prime = self.game.player(h)

            if i_prime is None:
                ps = self.game.chance_probabilities(h)
            elif i_prime == i:
                ps = self.reference_strategy_profile(h)
            else:
                ps = self._action_probabilities(h)

            k = np.random.choice(len(A), p=ps)
            a = A[k]
            h_prime = self.game.apply(h, a)
            p_prime = ps[k] * p
            u_prime = ps[k] * self._outcome_sampling(i, us, h_prime, p_prime)
            u += u_prime

            if i_prime == i:
                self.regret_minimizer(h)

                j = self.game.information_set(h)
                us[j] = np.zeros(len(A), dtype)
                us[j][k] = u_prime

        return u

    def _outcome_sampling2(self, player):
        us = {}

        self._outcome_sampling(player, us, self.game.root_node, 1)

        return us

    def sample(self, player):
        """Sample.

        :param player: Player.
        :return: Utilities.
        """
        self.next_sample_count += 1

        if self.reference_strategy_profile is None:
            us = self._external_sampling2(player)
        else:
            us = self._outcome_sampling2(player)

        return us

    def observe(self, utilities):
        """Observe utilities.

        :param utilities: Utilities.
        :return: ``None``.
        """
        self.sample_count = self.next_sample_count
        self.node_visit_count = self.next_node_visit_count

        for j, u in utilities.items():
            R = self.regret_minimizers[j]

            if R.next_strategy is None:
                R.output()

            R.observe(u)


@dataclass
class MonteCarloCounterfactualRegretMinimization(
        StochasticRegretMinimizer,
        ABC,
):
    """Class for Monte Carlo counterfactual regret minimization (MCCFR)."""
    regret_minimizer_type: type[ProbabilitySimplexRegretMinimizer] = (
        RegretMatching
    )
