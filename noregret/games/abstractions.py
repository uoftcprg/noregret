from abc import ABC, abstractmethod
from collections.abc import Callable
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any

from ordered_set import OrderedSet
from scipy.cluster.vq import vq

from noregret.games.black_box import BlackBoxGame


@dataclass
class Abstraction(BlackBoxGame, ABC):
    """Abstraction."""
    game: BlackBoxGame
    """Game."""

    @property
    @abstractmethod
    def is_perfect_recall(self):
        """Return whether the abstraction is perfect-recall.

        :return: Whether the abstraction is perfect-recall.
        """

    @abstractmethod
    def abstract_action(
            self,
            decision_point,
            action,
            abstract_decision_point,
            clustered_actions,
    ):
        """Abstract action.

        :param decision_point: Decision point.
        :param action: Action.
        :param abstract_decision_point: Abstract decision point.
        :param clustered_actions: Clustered (abstract) actions.
        :return: Abstracted action.
        """

    @abstractmethod
    def lift_action(
            self,
            decision_point,
            actions,
            abstract_decision_point,
            abstract_action,
    ):
        """Lift (abstract) action.

        :param decision_point: Decision point.
        :param actions: Actions.
        :param abstract_decision_point: Abstract decision point.
        :param abstract_action: Abstract action.
        :return: Lifted action.
        """

    def to_lifted_behavioral_form(self, abstract_strategy_profile):
        """Lift and convert (abstract) strategy profile to behavioral form.

        :param strategy_profile: Strategy profile.
        :return: Lifted behavioral-form strategy profile.
        """
        strategy_profile = {}
        hs = [(self.game.root_node, self.root_node)]

        while hs:
            h, h_star = hs.pop()
            i = self.game.player(h)
            i_star = self.player(h_star)

            if i != i_star:
                raise ValueError('unsupported game definition')

            A_j, h_primes = self.game.actions_and_children(h)

            if i is None:
                A_j_star, h_star_primes = self.actions_and_children(h_star)

                if A_j != A_j_star:
                    raise ValueError('unsupported game tree topology')
            else:
                j = self.game.information_set(h)
                j_star = self.information_set(h_star)
                _, clusters = h_star
                A_j_star2 = [
                    self.abstract_action(j, a, j_star, clusters) for a in A_j
                ]
                h_star_primes = list(
                    map(partial(self.apply, h_star), A_j_star2),
                )
                A_j_star = self.actions(h_star)
                p_stars = abstract_strategy_profile(h_star)
                ps = defaultdict(int)

                for a_star, p_star in zip(A_j_star, p_stars):
                    a = self.lift_action(j, A_j, j_star, a_star)
                    ps[a] += p_star

                for a in A_j:
                    strategy_profile[j, a] = ps[a]

            assert len(h_primes) == len(h_star_primes)

            hs.extend(zip(h_primes, h_star_primes))

        return strategy_profile


@dataclass
class ActionAbstraction(Abstraction, ABC):
    """Action abstraction."""

    @property
    def is_perfect_recall(self):
        return True

    @property
    def player_count(self):
        return self.game.player_count

    @property
    def is_zero_sum(self):
        return self.game.is_zero_sum

    @property
    def root_node(self):
        node = self.game.root_node

        return self._abstract_node(node)

    def _abstract_node(self, node):
        actions = self.game.actions(node)

        if self.game.player(node) is None:
            clusters = [OrderedSet([action]) for action in actions]
        else:
            decision_point = self.game.information_set(node)
            clusters = self.cluster_actions(decision_point, actions)

        return node, clusters

    def abstract_action(
            self,
            decision_point,
            action,
            abstract_decision_point,
            clustered_actions,
    ):
        np = self.kernel.numpy
        abstract_actions = [cluster[0] for cluster in clustered_actions]

        if action in abstract_actions:
            abstract_action = action
        elif abstract_actions:
            state = np.random.get_state()
            seed = self.seed(abstract_decision_point, abstract_actions)

            np.random.seed(seed)

            index = np.random.choice(len(abstract_actions))

            np.random.set_state(state)

            abstract_action = abstract_actions[index]
        else:
            abstract_action = None

        return abstract_action

    def lift_action(
            self,
            decision_point,
            actions,
            abstract_decision_point,
            abstract_action,
    ):
        np = self.kernel.numpy
        state = np.random.get_state()

        if abstract_action in actions:
            action = abstract_action
        elif actions:
            state = np.random.get_state()
            seed = self.seed(decision_point, actions)

            np.random.seed(seed)

            index = np.random.choice(len(actions))

            np.random.set_state(state)

            action = actions[index]
        else:
            action = None

        np.random.set_state(state)

        return action

    @abstractmethod
    def cluster_actions(self, decision_point, actions):
        pass

    def actions(self, node):
        _, clusters = node
        actions = OrderedSet()

        for cluster in clusters:
            actions.add(cluster[0])

        return actions

    def apply(self, node, action):
        node, _ = node
        node = self.game.apply(node, action)

        return self._abstract_node(node)

    def player(self, node):
        node, _ = node

        return self.game.player(node)

    def utility(self, node, player):
        node, _ = node

        return self.game.utility(node, player)

    def information_set(self, node):
        node, _ = node

        return self.game.information_set(node)

    def chance_probability(self, node, action):
        raise NotImplementedError

    def chance_probabilities(self, node):
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        node, clusters = node
        A = self.game.actions(node)
        ps = self.game.chance_probabilities(node)
        lookup = dict(zip(A, ps))
        ps = []

        for cluster in clusters:
            p = 0

            for a in cluster:
                p += lookup[a]

            ps.append(p)

        return np.array(ps, dtype)


@dataclass
class RandomActionAbstraction(ActionAbstraction, ABC):
    """Random action abstraction."""
    seed: Callable[[str, OrderedSet[str]], Any]
    """Seed."""
    k: Callable[[str, OrderedSet[str]], Any]
    """Number of clusters."""

    def cluster_actions(self, decision_point, actions):
        np = self.kernel.numpy
        k = self.k(decision_point, actions)

        if k is None or k >= len(actions):
            clusters = [OrderedSet([action]) for action in actions]
        else:
            clusters = [OrderedSet() for _ in range(k)]
            state = np.random.get_state()
            seed = self.seed(decision_point, actions)

            np.random.seed(seed)

            labels = np.random.randint(k, size=len(actions))

            np.random.set_state(state)

            for label, action in zip(labels, actions):
                clusters[label].add(action)

            clusters = list(filter(None, clusters))

        return clusters


@dataclass
class EmbeddingAbstraction(ActionAbstraction, ABC):
    """Embedding abstraction."""
    embed: Callable[[str, str], Any]
    """(Embedding) model."""
    k: Callable[[str, OrderedSet[str]], Any]
    """Number of clusters."""
    cluster: Callable[[str, int], Any]
    """Cluster."""

    def abstract_action(
            self,
            decision_point,
            action,
            abstract_decision_point,
            clustered_actions,
    ):
        np = self.kernel.numpy

        if clustered_actions:
            abstract_actions = [cluster[0] for cluster in clustered_actions]
            abstract_embeddings = list(
                map(
                    partial(self.embed, abstract_decision_point),
                    abstract_actions,
                ),
            )
            embedding = self.embed(decision_point, action)
            (index,), _ = vq(embedding[np.newaxis, :], abstract_embeddings)
            abstract_action = abstract_actions[index]
        else:
            abstract_action = None

        return abstract_action

    def lift_action(
            self,
            decision_point,
            actions,
            abstract_decision_point,
            abstract_action,
    ):
        np = self.kernel.numpy

        if actions:
            embeddings = list(
                map(partial(self.embed, decision_point), actions),
            )
            abstract_embedding = self.embed(
                abstract_decision_point,
                abstract_action,
            )
            (index,), _ = vq(abstract_embedding[np.newaxis, :], embeddings)
            action = actions[index]
        else:
            action = None

        return action

    def cluster_actions(self, decision_point, actions):
        np = self.kernel.numpy
        k = self.k(decision_point, actions)

        if k is None or k >= len(actions):
            clusters = [OrderedSet([action]) for action in actions]
        else:
            embeddings = list(
                map(partial(self.embed, decision_point), actions),
            )
            centroids = self.cluster(embeddings, k)
            labels, distances = vq(embeddings, centroids)
            clusters = [OrderedSet() for _ in range(k)]

            for i in np.argsort(distances, kind='stable'):
                clusters[labels[i]].add(actions[i])

            clusters = list(filter(None, clusters))

        return clusters
