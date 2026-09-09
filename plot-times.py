from argparse import ArgumentParser
from collections import defaultdict
from itertools import repeat
from pathlib import Path

from orjson import loads
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import iteration_times

NOREGRET_GPU = 'Ours (GPU)'
NOREGRET_CPU = 'Ours (CPU)'


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('node_count', type=int)
    parser.add_argument('branching_factors')
    parser.add_argument('noregret_gpu')
    parser.add_argument('noregret_cpu')
    parser.add_argument('figures', nargs='*', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    branching_factors = list(map(int, args.branching_factors.split()))
    data = defaultdict(list)

    for branching_factor in branching_factors:
        nB = args.node_count, branching_factor
        noregret_gpu = loads(open(args.noregret_gpu.format(*nB), 'rb').read())
        noregret_cpu = loads(open(args.noregret_cpu.format(*nB), 'rb').read())

        data[''].extend(repeat(NOREGRET_GPU, len(noregret_gpu['times'])))
        data['Branching factor'].extend(
            repeat(
                noregret_gpu['branching_factor'],
                len(noregret_gpu['times']),
            ),
        )
        data['Iteration time (s)'].extend(
            iteration_times(noregret_gpu['times']),
        )
        data[''].extend(repeat(NOREGRET_CPU, len(noregret_cpu['times'])))
        data['Branching factor'].extend(
            repeat(
                noregret_cpu['branching_factor'],
                len(noregret_cpu['times']),
            ),
        )
        data['Iteration time (s)'].extend(
            iteration_times(noregret_cpu['times']),
        )

    sns.set_context('notebook')
    sns.lineplot(
        data,
        x='Branching factor',
        y='Iteration time (s)',
        hue='',
        style='',
        markers=True,
        errorbar='sd',
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Iteration time vs. branching factor')
    plt.tight_layout()

    for figure in args.figures:
        plt.savefig(figure)


if __name__ == '__main__':
    main()
