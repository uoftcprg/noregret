from json import load, dump
from sys import stdin, stdout

from ordered_set import OrderedSet

NODE_TYPES = {'decision': 'DECISION_POINT', 'observation': 'OBSERVATION_POINT'}


def main():
    raw_game = load(stdin)
    raw_tfsdps = []

    for decision_problem in (
            raw_game['decision_problem_pl1'],
            raw_game['decision_problem_pl2'],
    ):
        raw_tfsdp = []
        parent_edges = OrderedSet()
        sequences = OrderedSet()

        for node in decision_problem:
            if (parent_edge := node['parent_edge']) is None:
                parent_edge = []

            raw_tfsdp.append(
                {
                    'parent_edge': parent_edge,
                    'node': {
                        'id': node['id'],
                        'type': NODE_TYPES[node['type']],
                    },
                },
            )

            parent_edges.add(tuple(parent_edge))

            if node['type'] == 'decision':
                sequences.update((node['id'], a) for a in node['actions'])

        for sequence in sequences - parent_edges:
            raw_tfsdp.append(
                {
                    'parent_edge': sequence,
                    'node': {'id': '', 'type': 'END_OF_THE_DECISION_PROCESS'},
                },
            )

        raw_tfsdps.append(raw_tfsdp)

    raw_utilities = []

    for utility in raw_game['utility_pl1']:
        raw_utilities.append(
            {
                'sequences': [
                    utility['sequence_pl1'],
                    utility['sequence_pl2'],
                ],
                'value': utility['value'],
            },
        )

    raw_game = {
        'tree_form_sequential_decision_processes': raw_tfsdps,
        'utilities': raw_utilities,
    }

    dump(raw_game, stdout)


if __name__ == '__main__':
    main()
