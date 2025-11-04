from collections import defaultdict
from json import dump, load
from sys import stdin, stdout


def main():
    raw_game = load(stdin)
    nodes = {}
    decision_points = set()
    observation_points = set()
    events = defaultdict(dict)

    for raw_tfsdp in raw_game['tree_form_sequential_decision_processes']:
        for raw_transition in raw_tfsdp:
            parent_edge = tuple(raw_transition['parent_edge'])
            node_id = raw_transition['node']['id']
            node_type = raw_transition['node']['type']

            if parent_edge:
                parent, event = parent_edge
                parent = nodes[parent]
                event = events[parent].setdefault(
                    event,
                    f'e{len(events[parent])}',
                )
                parent_edge = parent, event

            match node_type:
                case 'DECISION_POINT':
                    previous_node_id = node_id
                    node_id = f'd{len(decision_points)}'
                    nodes[previous_node_id] = node_id

                    decision_points.add(previous_node_id)
                case 'OBSERVATION_POINT':
                    previous_node_id = node_id
                    node_id = f'o{len(observation_points)}'
                    nodes[previous_node_id] = node_id

                    observation_points.add(previous_node_id)

            raw_transition['parent_edge'] = parent_edge
            raw_transition['node']['id'] = node_id

    for raw_utility in raw_game['utilities']:
        for i, sequence in enumerate(raw_utility['sequences']):
            if sequence:
                node, action = sequence
                node = nodes[node]
                action = events[node][action]
                sequence = node, action
                raw_utility['sequences'][i] = sequence

    dump(raw_game, stdout)


if __name__ == '__main__':
    main()
