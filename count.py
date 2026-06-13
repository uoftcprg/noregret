from argparse import ArgumentParser
from sys import stdout

from orjson import dumps
import noregret as nr


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('game')

    return parser.parse_args()


def count_nodes(game):
    node_count = 0

    def dfs(h):
        nonlocal node_count

        node_count += 1

        for h_prime in game.children(h):
            dfs(h_prime)

    dfs(game.root_node)

    return node_count


def main():
    args = parse_args()
    ker = nr.FPKer()
    game = nr.OpenSpielGame(ker, args.game)
    node_count = count_nodes(game)
    game = nr.to_efg(ker, game)
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

    stdout.buffer.write(dumps(data))


if __name__ == '__main__':
    main()
