from argparse import ArgumentParser
from itertools import count
from orjson import dumps
from pathlib import Path
from resource import getrusage, RUSAGE_SELF
from time import time

from LiteEFG import OpenSpielEnv
from pyspiel import load_game
from tqdm import tqdm
import noregret as nr


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('game')
    parser.add_argument('R_path')
    parser.add_argument('total_time', type=int)
    parser.add_argument('iteration_count', type=int)
    parser.add_argument('data', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    R_type = nr.import_object(args.R_path)
    graph = R_type.graph()
    game = load_game(args.game)

    initial_time = time()
    env = OpenSpielEnv(game)

    env.set_graph(graph)

    setup_time = time() - initial_time

    checkpoint = 1
    times = []
    exploitabilities = []
    pbar = tqdm(total=args.total_time)
    initial_time = time()
    delay = 0

    for t in count(1):
        graph.update_graph(env)
        env.update_strategy(graph.current_strategy())

        time_ = time() - initial_time - delay
        status = (
            time_ >= pbar.total
            and t >= args.iteration_count
        )

        if t == checkpoint or status:
            checkpoint *= 2
            sigma = graph.current_strategy()
            delay -= time()
            epsilon = sum(env.exploitability(sigma, 'avg-iterate'))
            delay += time()
        else:
            epsilon = None

        times.append(time_)

        if epsilon is not None:
            exploitabilities.append(epsilon)

        pbar.update(min(pbar.total, int(time_)) - pbar.n)

        if status:
            break

    ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    data = {
        'game': args.game,
        'R_path': args.R_path,
        'total_time': args.total_time,
        'iteration_count': args.iteration_count,
        'setup_time': setup_time,
        'times': times,
        'exploitabilities': exploitabilities,
        'ru_maxrss': ru_maxrss,
    }

    with open(args.data, 'wb') as file:
        file.write(dumps(data))


if __name__ == '__main__':
    main()
