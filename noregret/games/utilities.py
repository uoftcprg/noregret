from collections import defaultdict
from functools import partial
from itertools import starmap

from ordered_set import OrderedSet
from pyspiel import GameType, load_game
from scipy.sparse import lil_array

from noregret.games.normal_form.games import (
    NormalFormGame,
    TwoPlayerNormalFormGame,
    TwoPlayerZeroSumNormalFormGame,
)
from noregret.games.extensive_form.games import (
    ExtensiveFormGame,
    TwoPlayerExtensiveFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
)
from noregret.sequence_form_polytopes import SequenceFormPolytope


def _nfg2efg(game, decision_points):
    kernel = game.kernel
    np = kernel.numpy
    scipy = kernel.scipy
    dtype = kernel.data_type

    if isinstance(game, TwoPlayerZeroSumNormalFormGame):
        type_ = TwoPlayerZeroSumExtensiveFormGame
    elif isinstance(game, TwoPlayerNormalFormGame):
        type_ = TwoPlayerExtensiveFormGame
    else:
        type_ = ExtensiveFormGame

    d = game.dimensions

    if isinstance(game, TwoPlayerZeroSumNormalFormGame):
        payoffs = np.zeros(tuple(n + 1 for n in d), dtype)
        payoffs[tuple(slice(1, None) for _ in d)] = game.payoffs
    else:
        payoffs = np.zeros((game.player_count, *(n + 1 for n in d)), dtype)
        payoffs[:, *(slice(1, None) for _ in d)] = game.payoffs

    payoffs = scipy.sparse.csr_array(payoffs)
    sequence_form_polytopes = []

    for i, A_j in enumerate(game.actions):
        j = decision_points(i)
        sfp = SequenceFormPolytope(kernel, {j: A_j}, {j: None})

        sequence_form_polytopes.append(sfp)

    sequence_form_polytopes = tuple(sequence_form_polytopes)

    return type_(kernel, payoffs, sequence_form_polytopes)


def to_extensive_form(game, decision_points=str):
    """Convert a given game to an extensive-form game.

    :param game: Game.
    :param decision_points: Decision points, defaults to ``str''.
    :return: Extensive-form game.
    """
    if isinstance(game, NormalFormGame):
        game = _nfg2efg(game, decision_points)
    else:
        raise ValueError('unknown game')

    return game


def from_open_spiel(kernel, game):
    """Load a game from OpenSpiel.

    :param kernel: Kernel.
    :param game: Game in OpenSpiel.
    :return: Game.
    """
    dtype = kernel.data_type
    scipy = kernel.scipy
    game = load_game(game)
    player_count = game.num_players()
    actions = [defaultdict(OrderedSet) for _ in range(player_count)]
    parent_sequences = [{} for _ in range(player_count)]
    raw_payoffs = [defaultdict(int) for _ in range(player_count)]

    def dfs(state, chance_probability, sequences):
        if state.is_terminal():
            key = tuple(sequences)

            for i, u in enumerate(state.rewards()):
                raw_payoffs[i][key] += chance_probability * u
        elif state.is_chance_node():
            for a, p in state.chance_outcomes():
                dfs(state.child(a), p * chance_probability, sequences)
        else:
            i = state.current_player()
            j = state.information_state_string()
            p_j = sequences[i]
            parent_sequences[i][j] = p_j

            for a in state.legal_actions():
                next_state = state.child(a)
                a = state.action_to_string(a)
                next_sequences = sequences.copy()
                next_sequences[i] = j, a

                actions[i][j].add(a)
                dfs(next_state, chance_probability, next_sequences)

    dfs(game.new_initial_state(), 1, [None] * player_count)

    sequence_form_polytopes = tuple(
        starmap(
            partial(SequenceFormPolytope, kernel),
            zip(actions, parent_sequences),
        ),
    )
    dimensions = tuple(sfp.column_count for sfp in sequence_form_polytopes)

    if (
            player_count == 2
            and game.get_type().utility == GameType.Utility.ZERO_SUM
    ):
        type_ = TwoPlayerZeroSumExtensiveFormGame
        payoffs = lil_array(dimensions, dtype=dtype)

        for sequences, payoff in raw_payoffs[0].items():
            indices = []

            for sfp, sequence in zip(sequence_form_polytopes, sequences):
                indices.append(sfp.column(sequence))

            payoffs[tuple(indices)] = payoff

        payoffs = scipy.sparse.csr_array(payoffs)
    else:
        raise NotImplementedError

    return type_(kernel, payoffs, sequence_form_polytopes)
