from argparse import ArgumentParser
from collections import defaultdict
from itertools import repeat
from pathlib import Path

from orjson import loads
import matplotlib.pyplot as plt
import seaborn as sns

ROW_COUNT = 2
COLUMN_COUNT = 4
FIGURE_SIZE = 16, 8
GAMES = {
    'kuhn-poker': 'Kuhn poker',
    'leduc-poker': 'Leduc poker',
    'liars-dice': 'Liar\'s dice',
    'goofspiel-6': 'Goofspiel-6',
    'goofspiel-7': 'Goofspiel-7',
    'battleship-3x2-2-3': 'Battleship-3x2-2-3',
    'battleship-3x2-22-3': 'Battleship-3x2-22-3',
}
GAME_COUNT = len(GAMES)
GPU = 'NoRegret (GPU)'
CPU = 'NoRegret (CPU)'
CPP = 'OpenSpiel (C++)'
PYTHON = 'OpenSpiel (Python)'


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('total_time', type=int)
    parser.add_argument('gpu')
    parser.add_argument('cpu')
    parser.add_argument('cpp')
    parser.add_argument('python')
    parser.add_argument('figures', nargs='*', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    data = {}

    for game in GAMES:
        gpu = loads(open(args.gpu.format(game), 'rb').read())
        cpu = loads(open(args.cpu.format(game), 'rb').read())
        cpp = loads(open(args.cpp.format(game), 'rb').read())
        python = loads(open(args.python.format(game), 'rb').read())
        data[game] = defaultdict(list)

        data[game][''].extend(repeat(GPU, len(gpu['times'])))
        data[game]['Wall time (s)'].extend(gpu['times'])
        data[game]['Exploitability'].extend(gpu['exploitabilities'])
        data[game][''].extend(repeat(CPU, len(cpu['times'])))
        data[game]['Wall time (s)'].extend(cpu['times'])
        data[game]['Exploitability'].extend(cpu['exploitabilities'])
        data[game][''].extend(repeat(CPP, len(cpp['times'])))
        data[game]['Wall time (s)'].extend(cpp['times'])
        data[game]['Exploitability'].extend(cpp['exploitabilities'])
        data[game][''].extend(repeat(PYTHON, len(python['times'])))
        data[game]['Wall time (s)'].extend(python['times'])
        data[game]['Exploitability'].extend(python['exploitabilities'])

    sns.set_context('notebook')

    fig, axes = plt.subplots(ROW_COUNT, COLUMN_COUNT, figsize=FIGURE_SIZE)
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for i, (game, ax) in enumerate(zip(GAMES, axes)):
        legend = i == GAME_COUNT - 1

        sns.lineplot(
            data[game],
            x='Wall time (s)',
            y='Exploitability',
            hue='',
            style='',
            legend=legend,
            ax=ax,
        )
        ax.set_xlim((-args.total_time * 0.05, args.total_time * 1.05))
        ax.set_yscale('log')
        ax.set_title(GAMES[game])

        if legend:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

            ax.get_legend().remove()

    ax = axes[-1]

    sns.set_context('talk')
    ax.axis('off')
    ax.legend(
        legend_handles,
        legend_labels,
        loc='center',
        frameon=False,
    )
    fig.tight_layout()

    for figure in args.figures:
        fig.savefig(figure)


if __name__ == '__main__':
    main()
