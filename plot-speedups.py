from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from orjson import loads
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utilities import iteration_times

NOREGRET_CUDA = 'Ours (GPU)'
NOREGRET_FP = 'Ours (ST-CPU)'


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('node_count', type=int)
    parser.add_argument('branching_factors')
    parser.add_argument('noregret_cuda')
    parser.add_argument('noregret_fp')
    parser.add_argument('figures', nargs='*', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    branching_factors = list(map(int, args.branching_factors.split()))
    data = defaultdict(list)

    for branching_factor in branching_factors:
        nB = args.node_count, branching_factor
        noregret_cuda = loads(
            open(args.noregret_cuda.format(*nB), 'rb').read(),
        )
        noregret_fp = loads(open(args.noregret_fp.format(*nB), 'rb').read())
        speedup = (
            np.mean(iteration_times(noregret_fp['times']))
            / np.mean(iteration_times(noregret_cuda['times']))
        )

        data['Branching factor'].append(noregret_cuda['branching_factor'])
        data['Speedup (times)'].append(speedup)

    sns.set_context('notebook')
    sns.lineplot(
        data,
        x='Branching factor',
        y='Speedup (times)',
        style=True,
        markers=True,
        legend=False,
    )
    plt.xscale('log')
    plt.title('Speedup vs. branching factor')
    plt.tight_layout()

    for figure in args.figures:
        plt.savefig(figure)


if __name__ == '__main__':
    main()
