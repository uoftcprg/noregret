from argparse import ArgumentParser
from math import perm
from pathlib import Path

from orjson import dumps
import noregret as nr


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('game')
    parser.add_argument('count', type=Path)
    parser.add_argument('data', type=Path)

    return parser.parse_args()


def count_nodes(game_type):
    state = game_type.create_state(())

    def help_count_nodes(card_count, street_index, actions):
        if actions.endswith('f'):
            count = 1
        elif actions.endswith('c') and actions != 'c':
            count = 1

            if street_index + 1 != state.street_count:
                count += card_count * help_count_nodes(
                    card_count - 1,
                    street_index + 1,
                    '',
                )
        else:
            street = state.streets[street_index]
            A_j = ['c']

            if actions.endswith('r'):
                A_j.append('f')

            if (
                    actions.count('r')
                    < street.max_completion_betting_or_raising_count
            ):
                A_j.append('r')

            count = 1

            for a in A_j:
                count += help_count_nodes(
                    card_count,
                    street_index,
                    actions + a,
                )

        return count

    card_count = len(state.deck)
    n = state.player_count

    return 1 + perm(card_count, n) * help_count_nodes(card_count - n, 0, '')


def main():
    args = parse_args()
    ker = nr.FPKer()
    game_type = nr.import_object(args.game)
    game = nr.PokerKitGame(ker, game_type).to_extensive_form()
    node_count = count_nodes(game_type)
    row_sfp = game.row_sequence_form_polytope
    col_sfp = game.column_sequence_form_polytope
    decision_point_count = (
        len(row_sfp.decision_points)
        + len(col_sfp.decision_points)
    )
    action_count = (
        len(row_sfp.non_empty_sequences)
        + len(col_sfp.non_empty_sequences)
    )
    payoff_count = game.payoffs.count_nonzero().item()
    data = {
        'node_count': node_count,
        'decision_point_count': decision_point_count,
        'action_count': action_count,
        'payoff_count': payoff_count,
    }

    with open(args.count, 'wb') as file:
        file.write(dumps(data))

    with open(args.data, 'wb') as file:
        file.write(game.dumps())


if __name__ == '__main__':
    main()
