from argparse import ArgumentParser
from collections import defaultdict
from sys import stdout

from humanize import naturalsize
from orjson import loads
import pandas as pd

GAMES = {
    'kuhn-poker': 'Kuhn poker',
    'leduc-poker': 'Leduc poker',
    'liars-dice': 'Liar\'s dice',
    'goofspiel-6': 'Goofspiel-6',
    'goofspiel-7': 'Goofspiel-7',
    'battleship-3x2-2-3': 'Battleship-3x2-2-3',
    'battleship-3x2-22-3': 'Battleship-3x2-22-3',
}
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

    return parser.parse_args()


def main():
    args = parse_args()
    data = defaultdict(list)

    for game in GAMES:
        gpu = loads(open(args.gpu.format(game), 'rb').read())
        cpu = loads(open(args.cpu.format(game), 'rb').read())
        cpp = loads(open(args.cpp.format(game), 'rb').read())
        python = loads(open(args.python.format(game), 'rb').read())

        data['Game'].append(GAMES[game])
        data[GPU].append(naturalsize(gpu['ru_maxrss'] * 1024))
        data[CPU].append(naturalsize(cpu['ru_maxrss'] * 1024))
        data[CPP].append(naturalsize(cpp['ru_maxrss'] * 1024))
        data[PYTHON].append(naturalsize(python['ru_maxrss'] * 1024))
        data['CUDA'].append(naturalsize(gpu['used_bytes']))

    df = pd.DataFrame(data)

    df.to_latex(stdout)


if __name__ == '__main__':
    main()
