from argparse import ArgumentParser, BooleanOptionalAction
from itertools import count
from json import dump
from pathlib import Path
from resource import getrusage, RUSAGE_SELF
from sys import stdout
from time import time

from noregret.utilities import import_string
from tqdm import trange, tqdm
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('game_path', type=Path)
    parser.add_argument('game_import_string')
    parser.add_argument('regret_minimizer_import_string')
    parser.add_argument('total_time', type=int)
    parser.add_argument('iteration_count', type=int)
    parser.add_argument('-a', '--alternate', action=BooleanOptionalAction)
    parser.add_argument('-e', '--exploitabilities', type=Path)
    parser.add_argument('-v', '--values', type=Path)
    parser.add_argument('-n', '--game_name')

    return parser.parse_args()


def main():
    args = parse_args()
    memory_pool = cp.get_default_memory_pool()
    pinned_memory_pool = cp.get_default_pinned_memory_pool()
    game_type = import_string(args.game_import_string)

    with open(args.game_path) as file:
        game = game_type.load(file)

    if args.game_name:
        game_name = args.game_name
    else:
        game_name = args.game_path.stem

    regret_minimizer_type = import_string(args.regret_minimizer_import_string)
    row_tfsdp = game.row_tree_form_sequential_decision_process
    row_cfr = regret_minimizer_type(row_tfsdp)
    column_tfsdp = game.column_tree_form_sequential_decision_process
    column_cfr = regret_minimizer_type(column_tfsdp)
    average_row_strategy = row_cfr.average_strategy
    average_column_strategy = column_cfr.average_strategy
    checkpoint = 1
    iterations = []
    times = []
    wc_times = []
    exploitabilities = []
    values = []
    row_strategy = row_cfr.next_strategy()
    column_strategy = column_cfr.next_strategy()
    initial_time = time()
    pbar = tqdm(total=args.total_time)

    for iteration in count(1):
        begin_time = time()

        if args.alternate:
            row_utility = game.row_utility(column_strategy)

            row_cfr.observe_utility(row_utility)

            row_strategy = row_cfr.next_strategy()

            column_utility = game.column_utility(row_strategy)

            column_cfr.observe_utility(column_utility)

            column_strategy = column_cfr.next_strategy()
        else:
            row_utility = game.row_utility(column_strategy)
            column_utility = game.column_utility(row_strategy)

            row_cfr.observe_utility(row_utility)
            column_cfr.observe_utility(column_utility)

            row_strategy = row_cfr.next_strategy()
            column_strategy = column_cfr.next_strategy()

        end_time = time()
        time_ = end_time - begin_time
        wc_time = end_time - initial_time
        m = min(int(wc_time), args.total_time)
        pbar.update(m - pbar.n)
        average_row_strategy = row_cfr.average_strategy
        average_column_strategy = column_cfr.average_strategy
        average_strategies = (
            average_row_strategy.copy(),
            average_column_strategy.copy(),
        )
        terminate = (
            wc_time >= args.total_time
            and iteration >= args.iteration_count
        )

        if (iteration == checkpoint or terminate) and args.exploitabilities:
            exploitability = average_strategies
            checkpoint *= 2
        else:
            exploitability = None

        if args.values:
            value = game.row_value(*average_strategies)
        else:
            value = None

        iterations.append(iteration)
        times.append(time_)
        wc_times.append(wc_time)
        exploitabilities.append(exploitability)
        values.append(None if value is None else value.item())

        if terminate:
            break

    pbar.close()

    used_bytes = memory_pool.used_bytes()
    total_bytes = memory_pool.total_bytes()
    n_free_blocks = pinned_memory_pool.n_free_blocks()
    ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss

    for i in trange(len(exploitabilities)):
        exploitability = exploitabilities[i]

        if exploitability is not None:
            exploitabilities[i] = game.exploitability(*exploitability).item()

    data = {
        'Iteration': iterations,
        'Wall-clock time': wc_times,
        'Exploitability': exploitabilities,
        'Value': values,
    }
    df = pd.DataFrame(data)

    if args.exploitabilities:
        plt.clf()
        sns.lineplot(df, x='Iteration', y='Exploitability')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Exploitability of {game_name} in self-play')
        plt.savefig(args.exploitabilities)

    if args.values:
        plt.clf()
        sns.lineplot(df, x='Iteration', y='Value')
        plt.xscale('log')
        plt.title(f'Value of {game_name} in self-play')
        plt.savefig(args.values)

    row_sequences = row_tfsdp.sequences
    column_sequences = column_tfsdp.sequences
    data = {
        'game_path': str(args.game_path),
        'game_import_string': args.game_import_string,
        'regret_minimizer_import_string': args.regret_minimizer_import_string,
        'alternate': args.alternate,
        'total_time': args.total_time,
        'iteration_count': args.iteration_count,
        'iterations': iterations,
        'times': times,
        'wc_times': wc_times,
        'exploitabilities': exploitabilities,
        'values': values,
        'row_sequences': list(row_sequences),
        'average_row_strategy': average_row_strategy.tolist(),
        'column_sequences': list(column_sequences),
        'average_column_strategy': average_column_strategy.tolist(),
        'used_bytes': used_bytes,
        'total_bytes': total_bytes,
        'n_free_blocks': n_free_blocks,
        'ru_maxrss': ru_maxrss,
    }

    dump(data, stdout)


if __name__ == '__main__':
    main()
