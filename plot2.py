from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from orjson import loads
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
GPU = 'NoRegret (GPU)'
CPU = 'NoRegret (CPU)'
CPP = 'OpenSpiel (C++)'
PYTHON = 'OpenSpiel (Python)'


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('gpu')
    parser.add_argument('cpu')
    parser.add_argument('cpp')
    parser.add_argument('python')
    parser.add_argument('count')
    parser.add_argument('figures', nargs='*', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    time = defaultdict(list)
    space = defaultdict(list)

    for game in GAMES:
        gpu = loads(open(args.gpu.format(game), 'rb').read())
        cpu = loads(open(args.cpu.format(game), 'rb').read())
        cpp = loads(open(args.cpp.format(game), 'rb').read())
        python = loads(open(args.python.format(game), 'rb').read())
        count = loads(open(args.count.format(game), 'rb').read())
        n = count['node_count']

        time[''].append(GPU)
        time['Game size (# nodes)'].append(n)
        time['Iteration time (s)'].append(iteration_time(gpu['times'])[0])
        time[''].append(CPU)
        time['Game size (# nodes)'].append(n)
        time['Iteration time (s)'].append(iteration_time(cpu['times'])[0])
        time[''].append(CPP)
        time['Game size (# nodes)'].append(n)
        time['Iteration time (s)'].append(iteration_time(cpp['times'])[0])
        time[''].append(PYTHON)
        time['Game size (# nodes)'].append(n)
        time['Iteration time (s)'].append(iteration_time(python['times'])[0])

        space[''].append(GPU)
        space['Game size (# nodes)'].append(n)
        space['Memory usage (bytes)'].append(gpu['ru_maxrss'] * 1024)
        space['CUDA memory usage (bytes)'].append(gpu['used_bytes'])
        space[''].append(CPU)
        space['Game size (# nodes)'].append(n)
        space['Memory usage (bytes)'].append(cpu['ru_maxrss'] * 1024)
        space['CUDA memory usage (bytes)'].append(None)
        space[''].append(CPP)
        space['Game size (# nodes)'].append(n)
        space['Memory usage (bytes)'].append(cpp['ru_maxrss'] * 1024)
        space['CUDA memory usage (bytes)'].append(None)
        space[''].append(PYTHON)
        space['Game size (# nodes)'].append(n)
        space['Memory usage (bytes)'].append(python['ru_maxrss'] * 1024)
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
