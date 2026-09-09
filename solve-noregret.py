from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from time import time

from ordered_set import OrderedSet
from orjson import dumps
from tqdm import trange
import noregret as nr


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('node_count', type=int)
    parser.add_argument('branching_factor', type=int)
    parser.add_argument('ker_path')
    parser.add_argument('R_path')
    parser.add_argument('iteration_count', type=int)
    parser.add_argument('data', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    ker = nr.import_object(args.ker_path)()
    actions = defaultdict(OrderedSet)
    parent_sequences = {0: None}

    for h in trange(args.node_count):
        for a in range(1, args.branching_factor + 1):
            h_a = args.branching_factor * h + a

            if h_a < args.node_count:
                actions[h].add(a)

                assert h_a not in parent_sequences

                parent_sequences[h_a] = h, a
            else:
                break

    for h in parent_sequences.keys() - actions.keys():
        del parent_sequences[h]

    R_type = nr.import_object(args.R_path)
    sfp = nr.SequenceFormPolytope(ker, actions, parent_sequences)
    R = R_type(ker, sfp)
    us = ker.numpy.zeros(sfp.column_count)
    initial_time = time()
    times = []

    for _ in trange(args.iteration_count):
        R.output()
        R.observe(us)
        times.append(time() - initial_time)

    data = {
        'node_count': args.node_count,
        'branching_factor': args.branching_factor,
        'ker_path': args.ker_path,
        'R_path': args.R_path,
        'iteration_count': args.iteration_count,
        'times': times,
    }

    with open(args.data, 'wb') as file:
        file.write(dumps(data))


if __name__ == '__main__':
    main()
