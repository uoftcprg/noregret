from argparse import ArgumentParser
from itertools import count
from resource import getrusage, RUSAGE_SELF
from sys import stdout
from time import time

from orjson import dumps
from pyspiel import exploitability, load_game
from tqdm import tqdm
import noregret as nr


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('game')
    parser.add_argument('R_path')
    parser.add_argument('total_time', type=int)
    parser.add_argument('iteration_count', type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    game = load_game(args.game)
    R_type = nr.import_object(args.R_path)
    R = R_type(game)
    checkpoint = 1
    iterations = []
    times = []
    exploitabilities = []
    pbar = tqdm(total=args.total_time)
    initial_time = time()

    for t in count(1):
        R.evaluate_and_update_policy()

        time_ = time() - initial_time
        status = (
            time_ >= pbar.total
            and t >= args.iteration_count
        )

        if t == checkpoint or status:
            checkpoint *= 2

            if hasattr(R, 'tabular_average_policy'):
                sigma = R.tabular_average_policy()
            else:
                sigma = R.average_policy().to_dict()
        else:
            sigma = None

        iterations.append(t)
        times.append(time_)
        exploitabilities.append(sigma)
        pbar.update(min(pbar.total, int(time_)) - pbar.n)

        if status:
            break

    pbar.close()

    ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    indices = [i for i in range(len(exploitabilities)) if exploitabilities[i]]

    for i in tqdm(indices):
        sigma = exploitabilities[i]

        if sigma is not None:
            exploitabilities[i] = exploitability(game, sigma)

    data = {
        'game': args.game,
        'R_path': args.R_path,
        'total_time': args.total_time,
        'iteration_count': args.iteration_count,
        'iterations': iterations,
        'times': times,
        'exploitabilities': exploitabilities,
        'ru_maxrss': ru_maxrss,
    }

    stdout.buffer.write(dumps(data))


if __name__ == "__main__":
    main()
