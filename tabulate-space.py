from argparse import ArgumentParser
from collections import defaultdict

from humanize import naturalsize
from orjson import loads
from tqdm import tqdm
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
    parser.add_argument('table')

    return parser.parse_args()


def main():
    args = parse_args()
    data = defaultdict(list)

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

        data['Game'].append(GAMES[game])
        data[NOREGRET_GPU].append(
            naturalsize(noregret_gpu['ru_maxrss'] * 1024),
        )
        data[NOREGRET_CPU].append(
            naturalsize(noregret_cpu['ru_maxrss'] * 1024),
        )
        data[OPEN_SPIEL_CPP].append(
            naturalsize(open_spiel_cpp['ru_maxrss'] * 1024),
        )
        data[OPEN_SPIEL_PYTHON].append(
            naturalsize(open_spiel_python['ru_maxrss'] * 1024),
        )
        data[LITEEFG].append(naturalsize(liteefg['ru_maxrss'] * 1024))

        data['CUDA'].append(naturalsize(noregret_gpu['used_bytes']))

    df = pd.DataFrame(data)

    df.to_latex(args.table)


if __name__ == '__main__':
    main()
