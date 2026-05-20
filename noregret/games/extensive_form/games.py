"""Module for extensive-form games (EFGs)."""
from dataclasses import dataclass
from io import BytesIO

from ordered_set import OrderedSet
from orjson import dumps, loads
from scipy.sparse import load_npz, save_npz

from noregret.games.multilinear import (
    MultilinearGame,
    TwoPlayerMultilinearGame,
    TwoPlayerZeroSumMultilinearGame,
)
from noregret.kernels import Serializable
from noregret.sequence_form_polytopes import SequenceFormPolytope
from noregret.utilities import tuple_or_none


@dataclass
class ExtensiveFormGame(MultilinearGame, Serializable):
    """Extensive-form game (EFG).

    Every player optimizes over a sequence-form polytope.
    """
    sequence_form_polytopes: tuple[SequenceFormPolytope, ...]
    """Sequence-form polytopes."""

    @property
    def player_count(self):
        return len(self.sequence_form_polytopes)

    @property
    def dimensions(self):
        return tuple(sfp.column_count for sfp in self.sequence_form_polytopes)

    def best_response_value(self, player, *strategies):
        u = self.utility(player, *strategies)

        return self.sequence_form_polytopes[player].best_response_value(u)

    @classmethod
    def loads(cls, kernel, raw_data):

        def sfp(raw_sfp):
            actions = raw_sfp['actions']
            J = actions.keys()
            A = actions.values()
            actions = dict(zip(J, map(OrderedSet, A)))
            parent_sequences = raw_sfp['parent_sequences']
            J = parent_sequences.keys()
            sequences = parent_sequences.values()
            parent_sequences = dict(zip(J, map(tuple_or_none, sequences)))

            return SequenceFormPolytope(kernel, actions, parent_sequences)

        data = loads(raw_data)
        io = BytesIO(bytes.fromhex(data['payoffs']))
        payoffs = kernel.scipy.sparse.csr_array(load_npz(io))
        sfps = tuple(map(sfp, data['sequence_form_polytopes']))

        return cls(kernel, payoffs, sfps)

    def dumps(self):

        def raw_sfp(sfp):
            return {
                'actions': sfp.actions,
                'parent_sequences': sfp.parent_sequences,
            }

        io = BytesIO()
        sfps = self.sequence_form_polytopes

        save_npz(io, self.payoffs)

        data = {
            'payoffs': io.getvalue().hex(),
            'sequence_form_polytopes': list(map(raw_sfp, sfps)),
        }

        return dumps(data, list)


@dataclass
class TwoPlayerExtensiveFormGame(TwoPlayerMultilinearGame, ExtensiveFormGame):
    """Class for two-player (2p) extensive-form games (EFGs)."""

    @property
    def row_sequence_form_polytope(self):
        """Return the sequence-form polytope for the row player.

        :return: Sequence-form polytope for the row player.
        """
        return self.sequence_form_polytopes[0]

    @property
    def column_sequence_form_polytope(self):
        """Return the sequence-form polytope for the column player.

        :return: Sequence-form polytope for the column player.
        """
        return self.sequence_form_polytopes[1]

    @property
    def row_dimension(self):
        """Return the dimension for the row player.

        :return: Dimension for the row player.
        """
        return self.row_sequence_form_polytope.column_count

    @property
    def column_dimension(self):
        """Return the dimension for the column player.

        :return: Dimension for the column player.
        """
        return self.column_sequence_form_polytope.column_count

    def row_best_response_value(self, column_strategy):
        u = self.row_utility(column_strategy)

        return self.row_sequence_form_polytopes.best_response_value(u)

    def column_best_response_value(self, row_strategy):
        v = self.column_utility(row_strategy)

        return self.column_sequence_form_polytopes.best_response_value(v)


@dataclass
class TwoPlayerZeroSumExtensiveFormGame(
        TwoPlayerZeroSumMultilinearGame,
        TwoPlayerExtensiveFormGame,
):
    """Class for two-player zero-sum (2p0s) extensive-form games (EFGs)."""

    def _best_response_row_values(self, row_strategy, column_strategy):
        u, neg_v = self._row_utilities(row_strategy, column_strategy)
        u = self.row_sequence_form_polytope.best_response_value(u)
        neg_v = self.column_sequence_form_polytope.worst_response_value(neg_v)

        return u, neg_v
