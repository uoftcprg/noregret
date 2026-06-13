from argparse import ArgumentParser
from resource import getrusage, RUSAGE_SELF
from sys import stdout
from time import time

from orjson import dumps
from tqdm import tqdm
import cupy as cp
import noregret as nr


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('game')
    parser.add_argument('ker_path')
    parser.add_argument('R_path')
    parser.add_argument('total_time', type=int)
    parser.add_argument('iteration_count', type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    cpu_ker = nr.FPKer()
    game = nr.OpenSpielGame(cpu_ker, args.game)
    ker = nr.import_object(args.ker_path)()
    game = nr.to_efg(ker, game)
    R_type = nr.import_object(args.R_path)
    row_sfp = game.row_sequence_form_polytope
    R_row = R_type(ker, row_sfp)
    col_sfp = game.column_sequence_form_polytope
    R_col = R_type(ker, col_sfp)
    checkpoint = 1
    iterations = []
    times = []
    exploitabilities = []
    pbar = tqdm(total=args.total_time)
    initial_time = time()

    def update():
        nonlocal checkpoint

        t = R_row.iteration_count
        time_ = time() - initial_time
        status = (
            time_ >= pbar.total
            and t >= args.iteration_count
        )

        if t == checkpoint or status:
            checkpoint *= 2
            x_bar = R_row.average_strategy.copy()
            y_bar = R_col.average_strategy.copy()

            sigma = x_bar, y_bar
        else:
            sigma = None

        iterations.append(t)
        times.append(time_)
        exploitabilities.append(sigma)
        pbar.update(min(pbar.total, int(time_)) - pbar.n)

        return status

    nr.rm(
        game,
        R_row,
        R_col,
        alternation=True,
        iteration_count=None,
        update=update,
        progress_bar=None,
    )
    pbar.close()

    memory_pool = cp.get_default_memory_pool()
    used_bytes = memory_pool.used_bytes()
    total_bytes = memory_pool.total_bytes()
    pinned_memory_pool = cp.get_default_pinned_memory_pool()
    n_free_blocks = pinned_memory_pool.n_free_blocks()
    ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    indices = [i for i in range(len(exploitabilities)) if exploitabilities[i]]

    for i in tqdm(indices):
        sigma = exploitabilities[i]

        if sigma is not None:
            exploitabilities[i] = game.exploitability(*sigma).item()

    data = {
        'ker_path': args.ker_path,
        'game': args.game,
        'R_path': args.R_path,
        'total_time': args.total_time,
        'iteration_count': args.iteration_count,
        'iterations': iterations,
        'times': times,
        'exploitabilities': exploitabilities,
        'used_bytes': used_bytes,
        'total_bytes': total_bytes,
        'n_free_blocks': n_free_blocks,
        'ru_maxrss': ru_maxrss,
    }

    stdout.buffer.write(dumps(data))


if __name__ == '__main__':
    main()
