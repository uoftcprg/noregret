from argparse import ArgumentParser
from collections import defaultdict
from sys import stdout

from orjson import loads
from pint import UnitRegistry
from uncertainties import ufloat
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
GPU = 'NoRegret (GPU)'
CPU = 'NoRegret (CPU)'
CPP = 'OpenSpiel (C++)'
PYTHON = 'OpenSpiel (Python)'
UREG = UnitRegistry()


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('gpu')
    parser.add_argument('cpu')
    parser.add_argument('cpp')
    parser.add_argument('python')

    return parser.parse_args()


def time(mean, sem):
    q = ufloat(mean, sem) * UREG.second

    return f'{q:~#P}'


def main():
    args = parse_args()
    data = defaultdict(list)

    for game in GAMES:
        gpu = loads(open(args.gpu.format(game), 'rb').read())
        cpu = loads(open(args.cpu.format(game), 'rb').read())
        cpp = loads(open(args.cpp.format(game), 'rb').read())
        python = loads(open(args.python.format(game), 'rb').read())

        data['Game'].append(GAMES[game])
        data[GPU].append(time(*iteration_time(gpu['times'])))
        data[CPU].append(time(*iteration_time(cpu['times'])))
        data[CPP].append(time(*iteration_time(cpp['times'])))
        data[PYTHON].append(time(*iteration_time(python['times'])))

    df = pd.DataFrame(data)

    df.to_latex(stdout)


if __name__ == '__main__':
    main()
