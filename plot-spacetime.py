from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from orjson import loads
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import iteration_time

ROW_COUNT = 1
COLUMN_COUNT = 4
FIGURE_SIZE = 16, 4
GAMES = (
    'kuhn-poker',
    'leduc-poker',
    'liars-dice',
    'goofspiel-6',
    'goofspiel-7',
    'battleship-3x2-2-3',
    'battleship-3x2-22-3',
)
NOREGRET_GPU = 'Ours (GPU)'
NOREGRET_CPU = 'Ours (CPU)'
OPEN_SPIEL_CPP = 'OpenSpiel (C++)'
OPEN_SPIEL_PYTHON = 'OpenSpiel (Python)'
LITEEFG = 'LiteEFG'


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('noregret_gpu')
    parser.add_argument('noregret_cpu')
    parser.add_argument('open_spiel_cpp')
    parser.add_argument('open_spiel_python')
    parser.add_argument('liteefg')
    parser.add_argument('count')
    parser.add_argument('figures', nargs='*', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    time = defaultdict(list)
    space = defaultdict(list)

    for game in tqdm(GAMES):
        noregret_gpu = loads(open(args.noregret_gpu.format(game), 'rb').read())
        noregret_cpu = loads(open(args.noregret_cpu.format(game), 'rb').read())
        open_spiel_cpp = loads(
            open(args.open_spiel_cpp.format(game), 'rb').read(),
        )
        open_spiel_python = loads(
            open(args.open_spiel_python.format(game), 'rb').read(),
        )
        liteefg = loads(open(args.liteefg.format(game), 'rb').read())
        count = loads(open(args.count.format(game), 'rb').read())
        n = count['node_count']

        time[''].append(NOREGRET_GPU)
        time['Game size (# nodes)'].append(n)
        time['Iteration time (s)'].append(
            iteration_time(noregret_gpu['times'])[0],
        )
        time[''].append(NOREGRET_CPU)
        time['Game size (# nodes)'].append(n)
        time['Iteration time (s)'].append(
            iteration_time(noregret_cpu['times'])[0],
        )
        time[''].append(OPEN_SPIEL_CPP)
        time['Game size (# nodes)'].append(n)
        time['Iteration time (s)'].append(
            iteration_time(open_spiel_cpp['times'])[0],
        )
        time[''].append(OPEN_SPIEL_PYTHON)
        time['Game size (# nodes)'].append(n)
        time['Iteration time (s)'].append(
            iteration_time(open_spiel_python['times'])[0],
        )
        time[''].append(LITEEFG)
        time['Game size (# nodes)'].append(n)
        time['Iteration time (s)'].append(iteration_time(liteefg['times'])[0])

        space[''].append(NOREGRET_GPU)
        space['Game size (# nodes)'].append(n)
        space['Memory usage (bytes)'].append(noregret_gpu['ru_maxrss'] * 1024)
        space['CUDA memory usage (bytes)'].append(noregret_gpu['used_bytes'])
        space[''].append(NOREGRET_CPU)
        space['Game size (# nodes)'].append(n)
        space['Memory usage (bytes)'].append(noregret_cpu['ru_maxrss'] * 1024)
        space['CUDA memory usage (bytes)'].append(None)
        space[''].append(OPEN_SPIEL_CPP)
        space['Game size (# nodes)'].append(n)
        space['Memory usage (bytes)'].append(
            open_spiel_cpp['ru_maxrss'] * 1024,
        )
        space['CUDA memory usage (bytes)'].append(None)
        space[''].append(OPEN_SPIEL_PYTHON)
        space['Game size (# nodes)'].append(n)
        space['Memory usage (bytes)'].append(
            open_spiel_python['ru_maxrss'] * 1024,
        )
        space['CUDA memory usage (bytes)'].append(None)
        space[''].append(LITEEFG)
        space['Game size (# nodes)'].append(n)
        space['Memory usage (bytes)'].append(liteefg['ru_maxrss'] * 1024)
        space['CUDA memory usage (bytes)'].append(None)

    fig, axes = plt.subplots(ROW_COUNT, COLUMN_COUNT, figsize=FIGURE_SIZE)
    axes = axes.flatten()

    sns.set_context('notebook')
    sns.lineplot(
        time,
        x='Game size (# nodes)',
        y='Iteration time (s)',
        hue='',
        style='',
        markers=True,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title('Iteration time')
    sns.lineplot(
        space,
        x='Game size (# nodes)',
        y='Memory usage (bytes)',
        hue='',
        style='',
        markers=True,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_title('Memory usage')
    sns.lineplot(
        space,
        x='Game size (# nodes)',
        y='CUDA memory usage (bytes)',
        hue='',
        style='',
        markers=True,
        ax=axes[2],
    )
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_title('CUDA memory usage')

    legend_handles, legend_labels = axes[2].get_legend_handles_labels()

    axes[2].get_legend().remove()
    sns.set_context('talk')
    axes[3].axis('off')
    axes[3].legend(
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
