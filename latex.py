from argparse import ArgumentParser
from collections import defaultdict
from sys import stdout

from humanize import scientific
from orjson import loads
import pandas as pd

from utilities import iteration_time

GAMES = {
    'kuhn-poker': 'Kuhn poker',
    'leduc-poker': 'Leduc poker',
    'liars-dice': 'Liar\'s dice',
    'goofspiel-6': 'Goofspiel-6',
    'goofspiel-7': 'Goofspiel-7',
    'battleship-3x2-2-3': 'Battleship-3x2-2-3',
    'battleship-3x2-22-3': 'Battleship-3x2-22-3',
}
CPP = 'OpenSpiel (C++)'
PYTHON = 'OpenSpiel (Python)'


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('gpu')
    parser.add_argument('cpp')
    parser.add_argument('python')
    parser.add_argument('count')

    return parser.parse_args()


def scientific2(value):
    if value < 1:
        value = -1 / value

    return f'{value:.2g}' if -100 < value < 100 else scientific(value, 1)


def main():
    args = parse_args()
    data = defaultdict(list)

    for game in GAMES:
        gpu = loads(open(args.gpu.format(game), 'rb').read())
        cpp = loads(open(args.cpp.format(game), 'rb').read())
        python = loads(open(args.python.format(game), 'rb').read())
        count = loads(open(args.count.format(game), 'rb').read())
        n = count['node_count']
        gpu = iteration_time(gpu['times'])[0]
        cpp = iteration_time(cpp['times'])[0] / gpu
        python = iteration_time(python['times'])[0] / gpu

        data['Game'].append(GAMES[game])
        data['# nodes'].append(scientific2(n))
        data[CPP].append(scientific2(cpp))
        data[PYTHON].append(scientific2(python))

    df = pd.DataFrame(data)

    df.to_latex(stdout)


if __name__ == '__main__':
    main()
