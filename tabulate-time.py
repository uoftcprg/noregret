from argparse import ArgumentParser
from collections import defaultdict

from orjson import loads
from pint import UnitRegistry
from tqdm import tqdm
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
NOREGRET_CUDA = 'Ours (GPU)'
NOREGRET_MKL = 'Ours (MT-CPU)'
NOREGRET_FP = 'Ours (ST-CPU)'
OPEN_SPIEL_CPP = 'OpenSpiel (C++)'
OPEN_SPIEL_PYTHON = 'OpenSpiel (Python)'
LITEEFG = 'LiteEFG'
UREG = UnitRegistry()


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('noregret_cuda')
    parser.add_argument('noregret_mkl')
    parser.add_argument('noregret_fp')
    parser.add_argument('open_spiel_cpp')
    parser.add_argument('open_spiel_python')
    parser.add_argument('liteefg')
    parser.add_argument('table')

    return parser.parse_args()


def time(mean, sem):
    q = ufloat(mean, sem) * UREG.second

    return f'{q:.1u~#P}'


def main():
    args = parse_args()
    data = defaultdict(list)

    for game in tqdm(GAMES):
        noregret_cuda = loads(
            open(args.noregret_cuda.format(game), 'rb').read(),
        )
        noregret_mkl = loads(open(args.noregret_mkl.format(game), 'rb').read())
        noregret_fp = loads(open(args.noregret_fp.format(game), 'rb').read())
        open_spiel_cpp = loads(
            open(args.open_spiel_cpp.format(game), 'rb').read(),
        )
        open_spiel_python = loads(
            open(args.open_spiel_python.format(game), 'rb').read(),
        )
        liteefg = loads(open(args.liteefg.format(game), 'rb').read())

        data['Game'].append(GAMES[game])
        data[NOREGRET_CUDA].append(
            time(*iteration_time(noregret_cuda['times'])),
        )
        data[NOREGRET_MKL].append(time(*iteration_time(noregret_mkl['times'])))
        data[NOREGRET_FP].append(time(*iteration_time(noregret_fp['times'])))
        data[OPEN_SPIEL_CPP].append(
            time(*iteration_time(open_spiel_cpp['times'])),
        )
        data[OPEN_SPIEL_PYTHON].append(
            time(*iteration_time(open_spiel_python['times'])),
        )
        data[LITEEFG].append(time(*iteration_time(liteefg['times'])))

    df = pd.DataFrame(data)

    df.to_latex(args.table)


if __name__ == '__main__':
    main()
