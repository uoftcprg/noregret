from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from itertools import count, permutations
from math import factorial
from typing import Any

from scipy.sparse import lil_array
import numpy as np

from noregret.utilities import (
    Serializable,
    split,
    TreeFormSequentialDecisionProcess,
)


@dataclass
class Game(ABC):
    """Game."""

    def __post_init__(self):
        self._verify()

    def _verify(self, **kwargs):
        pass

    @property
    @abstractmethod
    def player_count(self):
        pass

    @abstractmethod
    def dimension(self, player):
        pass

    @property
    def dimensions(self):
        return np.array(list(map(self.dimension, range(self.player_count))))

    @abstractmethod
    def utility(self, player, *opponent_strategies):
        pass

    @abstractmethod
    def value(self, player, *strategies):
        pass

    def values(self, *strategies):
        return np.array(
            [self.value(i, *strategies) for i in range(self.player_count)],
        )

    @abstractmethod
    def correlated_value(self, player, *strategies):
        pass

    def correlated_values(self, *strategies):
        return np.array(
            [
                self.correlated_value(i, *strategies)
                for i in range(self.player_count)
            ],
        )

    @abstractmethod
    def best_response(self, player, *opponent_strategies):
        pass

    def nash_gap(self, *strategies):
        gap = 0

        for i, value in enumerate(self.values(*strategies)):
            opponent_strategies = strategies[:i] + strategies[i + 1:]
            _, br_value = self.best_response(i, *opponent_strategies)
            gap += br_value - value

        return gap

    def cce_gap(self, *strategies):
        average_strategies = list(map(partial(np.mean, axis=0), strategies))
        gap = 0

        for i, value in enumerate(self.correlated_values(*strategies)):
            average_opponent_strategies = (
                average_strategies[:i] + average_strategies[i + 1:]
            )
            _, br_value = self.best_response(i, *average_opponent_strategies)
            gap += br_value - value

        return gap


@dataclass
class TwoPlayerGame(Game, ABC):
    """Two-player (2p) game.

    Row and column players are of indices 0 and 1, respectively.
    """

    def _verify(self, **kwargs):
        super()._verify(**kwargs)

        if self.player_count != 2:
            raise ValueError('number of players not 2')

    @property
    @abstractmethod
    def row_utilities(self):
        pass

    @property
    @abstractmethod
    def column_utilities(self):
        pass

    def dimension(self, player):
        match player:
            case 0:
                dimension = self.row_dimension
            case 1:
                dimension = self.column_dimension
            case _:
                raise ValueError(f'Player {player} does not exist')

        return dimension

    @property
    def row_dimension(self):
        return self.row_utilities.shape[0]

    @property
    def column_dimension(self):
        return self.row_utilities.shape[1]

    def utility(self, player, opponent_strategy):
        match player:
            case 0:
                utility = self.row_utility(opponent_strategy)
            case 1:
                utility = self.column_utility(opponent_strategy)
            case _:
                raise ValueError(f'Player {player} does not exist')

        return utility

    def row_utility(self, column_strategy):
        return self.row_utilities @ column_strategy

    def column_utility(self, row_strategy):
        return row_strategy @ self.column_utilities

    def value(self, player, row_strategy, column_strategy):
        match player:
            case 0:
                value = self.row_value(row_strategy, column_strategy)
            case 1:
                value = self.column_value(row_strategy, column_strategy)
            case _:
                raise ValueError(f'Player {player} does not exist')

        return value

    def row_value(self, row_strategy, column_strategy):
        return row_strategy @ self.row_utilities @ column_strategy

    def column_value(self, row_strategy, column_strategy):
        return row_strategy @ self.column_utilities @ column_strategy

    def correlated_value(self, player, row_strategies, column_strategies):
        match player:
            case 0:
                value = self.correlated_row_value(
                    row_strategies,
                    column_strategies,
                )
            case 1:
                value = self.correlated_column_value(
                    row_strategies,
                    column_strategies,
                )
            case _:
                raise ValueError(f'Player {player} does not exist')

        return value

    def correlated_row_value(self, row_strategies, column_strategies):
        return (
            row_strategies @ self.row_utilities * column_strategies
        ).sum(1).mean()

    def correlated_column_value(self, row_strategies, column_strategies):
        return (
            row_strategies @ self.column_utilities * column_strategies
        ).sum(1).mean()

    def best_response(self, player, opponent_strategy):
        match player:
            case 0:
                best_response = self.row_best_response(opponent_strategy)
            case 1:
                best_response = self.column_best_response(opponent_strategy)
            case _:
                raise ValueError(f'Player {player} does not exist')

        return best_response

    @abstractmethod
    def row_best_response(self, column_strategy):
        pass

    @abstractmethod
    def column_best_response(self, row_strategy):
        pass


@dataclass
class TwoPlayerZeroSumGame(TwoPlayerGame, ABC):
    """Two-player zero-sum (2p0s) game."""

    @property
    def column_utilities(self):
        return -self.row_utilities

    def values(self, row_strategy, column_strategy):
        value = self.row_value(row_strategy, column_strategy)

        return np.array((value, -value))

    def correlated_values(self, row_strategies, column_strategies):
        value = self.correlated_row_value(row_strategies, column_strategies)

        return np.array((value, -value))

    def nash_gap(self, row_strategy, column_strategy):
        _, row_best_response_value = self.row_best_response(column_strategy)
        _, column_best_response_value = self.column_best_response(row_strategy)

        return row_best_response_value + column_best_response_value

    def exploitability(self, row_strategy, column_strategy):
        return self.nash_gap(row_strategy, column_strategy) / 2


@dataclass
class NormalFormGame(Serializable, Game):
    """Normal-form game.

    Each player optimizes over the probability simplex.
    """

    @classmethod
    def deserialize(cls, raw_data):
        return cls(raw_data['actions'], np.array(raw_data['utilities']))

    actions: Any
    utilities: Any
    indices: Any = field(init=False, default_factory=list)

    def __post_init__(self):
        super().__post_init__()

        for i, actions in enumerate(self.actions):
            self.indices.append(dict(zip(actions, count())))

    def _verify(self, *, utilities_shape=None, **kwargs):
        super()._verify(**kwargs)

        if utilities_shape is None:
            utilities_shape = (*map(len, self.actions), self.player_count)

        if self.utilities.shape != utilities_shape:
            raise ValueError('utilities do not match actions and players')

    @property
    def player_count(self):
        return len(self.actions)

    def utility(self, player, *opponent_strategies):
        raise NotImplementedError

    def value(self, player, *strategies):
        raise NotImplementedError

    def correlated_value(self, player, *strategies):
        raise NotImplementedError

    def best_response(self, player, *opponent_strategies):
        raise NotImplementedError

    def serialize(self):
        return {'actions': self.actions, 'utilities': self.utilities.tolist()}


@dataclass
class TwoPlayerNormalFormGame(TwoPlayerGame, NormalFormGame):
    """Two-player (2p) normal-form game."""

    @property
    def row_actions(self):
        return self.actions[0]

    @property
    def column_actions(self):
        return self.actions[1]

    @property
    def row_indices(self):
        return self.indices[0]

    @property
    def column_indices(self):
        return self.indices[1]

    @property
    def row_utilities(self):
        return self.utilities[:, :, 0]

    @property
    def column_utilities(self):
        return self.utilities[:, :, 1]

    def row_best_response(self, column_strategy):
        strategy = np.zeros(len(self.row_actions))
        utility = self.row_utility(column_strategy)
        index = utility.argmax()
        strategy[index] = 1

        return strategy, utility[index]

    def column_best_response(self, row_strategy):
        strategy = np.zeros(len(self.column_actions))
        utility = self.column_utility(row_strategy)
        index = utility.argmax()
        strategy[index] = 1

        return strategy, utility[index]


@dataclass
class TwoPlayerZeroSumNormalFormGame(
        TwoPlayerZeroSumGame,
        TwoPlayerNormalFormGame,
):
    """Two-player zero-sum (2p0s) normal-form game.

    The utility matrix is from the viewpoint of the row player.
    """

    def _verify(self, **kwargs):
        super()._verify(
            **kwargs,
            utilities_shape=(len(self.row_actions), len(self.column_actions)),
        )

    @property
    def row_utilities(self):
        return self.utilities


@dataclass
class ExtensiveFormGame(Serializable, Game):
    """Extensive-form game (EFG).

    Each player optimizes over the sequence-form polytope.
    """

    @classmethod
    def deserialize(cls, raw_data):
        raise NotImplementedError

    tree_form_sequential_decision_processes: Any
    utilities: Any

    @property
    def player_count(self):
        return len(self.tree_form_sequential_decision_processes)

    def utility(self, player, *opponent_strategies):
        raise NotImplementedError

    def value(self, player, *strategies):
        raise NotImplementedError

    def correlated_value(self, player, *strategies):
        raise NotImplementedError

    def best_response(self, player, *opponent_strategies):
        raise NotImplementedError

    def serialize(self):
        raise NotImplementedError


@dataclass
class TwoPlayerExtensiveFormGame(TwoPlayerGame, ExtensiveFormGame):
    """Two-player (2p) extensive-form game (EFG)."""

    @classmethod
    def deserialize(cls, raw_data):
        tfsdps = TreeFormSequentialDecisionProcess.deserialize_all(
            raw_data['tree_form_sequential_decision_processes'],
        )
        shape = tuple(len(tfsdp.sequences) for tfsdp in tfsdps)
        row_utilities = lil_array(shape)
        column_utilities = lil_array(shape)

        for raw_utility in raw_data['utilities']:
            if len(raw_utility['values']) != 2:
                raise ValueError('utility is not of a 2-player game')

            indices = []

            for tfsdp, sequence in zip(tfsdps, raw_utility['sequences']):
                sequence = tuple(sequence)

                indices.append(tfsdp.indices[sequence])

            indices = tuple(indices)
            row_utilities[indices] = raw_utility['values'][0]
            column_utilities[indices] = raw_utility['values'][1]

        return cls(tfsdps, [row_utilities.tocsr(), column_utilities.tocsr()])

    def _verify(self, **kwargs):
        super()._verify(**kwargs)

        if not (
                self.row_utilities.shape
                == self.column_utilities.shape
                == (len(self.row_sequences), len(self.column_sequences))
        ):
            raise ValueError('utilities do not match sequences')

    @property
    def row_tree_form_sequential_decision_process(self):
        return self.tree_form_sequential_decision_processes[0]

    @property
    def column_tree_form_sequential_decision_process(self):
        return self.tree_form_sequential_decision_processes[1]

    @property
    def row_sequences(self):
        return self.row_tree_form_sequential_decision_process.sequences

    @property
    def column_sequences(self):
        return self.column_tree_form_sequential_decision_process.sequences

    @property
    def row_indices(self):
        return self.row_tree_form_sequential_decision_process.indices

    @property
    def column_indices(self):
        return self.column_tree_form_sequential_decision_process.indices

    @property
    def row_utilities(self):
        return self.utilities[0]

    @property
    def column_utilities(self):
        return self.utilities[1]

    def row_best_response(self, column_strategy):
        best_response = (
            self
            .row_tree_form_sequential_decision_process
            .sequence_form_best_response(self.row_utility(column_strategy))
        )

        return best_response

    def column_best_response(self, row_strategy):
        best_response = (
            self
            .column_tree_form_sequential_decision_process
            .sequence_form_best_response(self.column_utility(row_strategy))
        )

        return best_response

    def serialize(self):
        tfsdps = self.tree_form_sequential_decision_processes
        raw_tfsdps = [tfsdp.to_list() for tfsdp in tfsdps]
        raw_utilities = []
        abs_utility_sums = abs(self.row_utilities) + abs(self.column_utilities)

        for indices in zip(*abs_utility_sums.nonzero()):
            sequences = []

            for tfsdp, index in zip(tfsdps, indices):
                sequences.append(tfsdp.sequences[index])

            row_value = self.row_utilities[indices].item()
            column_value = self.column_utilities[indices].item()
            values = row_value, column_value

            raw_utilities.append({'sequences': sequences, 'values': values})

        return {
            'tree_form_sequential_decision_processes': raw_tfsdps,
            'utilities': raw_utilities,
        }


@dataclass
class TwoPlayerZeroSumExtensiveFormGame(
        TwoPlayerZeroSumGame,
        TwoPlayerExtensiveFormGame,
):
    """Two-player zero-sum (2p0s) extensive-form game (EFG).

    The utility matrix is from the viewpoint of the row player.
    """

    @classmethod
    def deserialize(cls, raw_data):
        tfsdps = TreeFormSequentialDecisionProcess.deserialize_all(
            raw_data['tree_form_sequential_decision_processes'],
        )
        shape = tuple(len(tfsdp.sequences) for tfsdp in tfsdps)
        utilities = lil_array(shape)

        for raw_utility in raw_data['utilities']:
            indices = []

            for tfsdp, sequence in zip(tfsdps, raw_utility['sequences']):
                sequence = tuple(sequence)

                indices.append(tfsdp.indices[sequence])

            indices = tuple(indices)
            utilities[indices] = raw_utility['value']

        return cls(tfsdps, utilities.tocsr())

    @property
    def row_utilities(self):
        return self.utilities

    def serialize(self):
        tfsdps = self.tree_form_sequential_decision_processes
        raw_tfsdps = [tfsdp.to_list() for tfsdp in tfsdps]
        raw_utilities = []

        for indices in zip(*self.utilities.nonzero()):
            sequences = []

            for tfsdp, index in zip(tfsdps, indices):
                sequences.append(tfsdp.sequences[index])

            value = self.utilities[indices].item()

            raw_utilities.append({'sequences': sequences, 'value': value})

        return {
            'tree_form_sequential_decision_processes': raw_tfsdps,
            'utilities': raw_utilities,
        }


@dataclass
class SymmetrizedGame(Game):
    """Symmetrized game.

    Each player optimizes over the cartesian product of probability
    simplices.
    """

    game: Any

    @property
    def player_count(self):
        return self.game.player_count

    def dimension(self, player):
        return sum(self.game.dimensions)

    def utility(self, player, *opponent_strategies):
        strategies = []

        for opponent_strategy in opponent_strategies:
            strategies.append(split(opponent_strategy, self.game.dimensions))

        strategies.insert(player, None)

        utilities = [0] * self.player_count

        for permutation in permutations(range(self.player_count)):
            utilities[permutation.index(player)] += self.game.utility(
                permutation.index(player),
                *(
                    strategies[permutation[i]][i]
                    for i in range(self.player_count)
                    if permutation[i] != player
                ),
            )

        utility = np.concatenate(utilities)
        utility /= factorial(self.player_count)

        return utility

    def value(self, player, *strategies):
        strategies = list(strategies)

        for i in range(self.player_count):
            strategies[i] = split(strategies[i], self.game.dimensions)

        value = 0

        for permutation in permutations(range(self.player_count)):
            value += self.game.value(
                permutation.index(player),
                *(
                    strategies[permutation[i]][i]
                    for i in range(self.player_count)
                ),
            )

        value /= factorial(self.player_count)

        return value

    def correlated_value(self, player, *strategies):
        raise NotImplementedError

    def best_response(self, player, *opponent_strategies):
        raise NotImplementedError
