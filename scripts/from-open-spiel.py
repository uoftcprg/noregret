from collections import defaultdict
from json import dump
from sys import argv, stdout
from warnings import warn

from ordered_set import OrderedSet
from pyspiel import GameType, load_game, SpielError
import numpy as np

GAME = argv[1]


def main():
    game = load_game(GAME)
    player_count = game.num_players()
    children = [{(): OrderedSet()} for _ in range(player_count)]
    utilities = defaultdict(int)

    def dfs(state, chance_probability, sequences):
        if state.is_terminal():
            utilities[tuple(sequences)] += (
                chance_probability * np.array(state.rewards())
            )
        elif state.is_chance_node():
            for action, probability in state.chance_outcomes():
                child = state.child(action)

                dfs(child, probability * chance_probability, sequences)
        else:
            try:
                infoset = state.information_state_string()
            except SpielError:
                warn('state as information set for rational player.')

                infoset = str(state)

            player = state.current_player()
            parent_sequence = sequences[player]

            children[player][parent_sequence].add(infoset)

            for action in state.legal_actions():
                child = state.child(action)
                action = state.action_to_string(action)
                sequence = infoset, action
                child_sequences = sequences.copy()
                child_sequences[player] = sequence

                children[player].setdefault(sequence, OrderedSet())
                dfs(child, chance_probability, child_sequences)

    dfs(game.new_initial_state(), 1, [()] * player_count)

    raw_tfsdps = [[] for _ in range(player_count)]
    observation_point_count = 0

    for player, raw_tfsdp in enumerate(raw_tfsdps):
        for sequence, infosets in children[player].items():
            if not infosets:
                raw_tfsdp.append(
                    {
                        'parent_edge': sequence,
                        'node': {
                            'id': '',
                            'type': 'END_OF_THE_DECISION_PROCESS',
                        },
                    },
                )
            elif len(infosets) == 1:
                raw_tfsdp.append(
                    {
                        'parent_edge': sequence,
                        'node': {'id': infosets[0], 'type': 'DECISION_POINT'},
                    },
                )
            else:
                i = observation_point_count
                observation_point_count += 1

                raw_tfsdp.append(
                    {
                        'parent_edge': sequence,
                        'node': {'id': f'o{i}', 'type': 'OBSERVATION_POINT'},
                    },
                )

                for j, infoset in enumerate(infosets):
                    raw_tfsdp.append(
                        {
                            'parent_edge': (f'o{i}', f'e{j}'),
                            'node': {'id': infoset, 'type': 'DECISION_POINT'},
                        },
                    )

    raw_utilities = [
        {'sequences': sequences, 'values': value.tolist()}
        for sequences, value in utilities.items()
    ]

    if (
            player_count == 2
            and game.get_type().utility == GameType.Utility.ZERO_SUM
    ):
        for raw_utility in raw_utilities:
            value, _ = raw_utility.pop('values')

            raw_utility['value'] = value

    raw_game = {
        'tree_form_sequential_decision_processes': raw_tfsdps,
        'utilities': raw_utilities,
    }

    dump(raw_game, stdout)


if __name__ == '__main__':
    main()
