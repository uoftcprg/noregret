from argparse import ArgumentParser
from collections import defaultdict

from humanize import scientific
from orjson import loads
from pint import UnitRegistry
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
NOREGRET_CUDA = 'Ours (GPU)'
LITEEFG = 'LiteEFG'
UREG = UnitRegistry()


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('noregret_cuda')
    parser.add_argument('liteefg')
    parser.add_argument('count')
    parser.add_argument('table')

    return parser.parse_args()


def scientific2(value):
    if value is None:
        return 'N/A'

    if value < 1:
        value = -1 / value

    return f'{value:.2g}' if -100 < value < 100 else scientific(value, 1)


def time(value):
    return f'{value * UREG.second:.1f~#P}'


def main():
    args = parse_args()
    data = defaultdict(list)

    for game in tqdm(GAMES):
        noregret_cuda = loads(
            open(args.noregret_cuda.format(game), 'rb').read(),
        )
        liteefg = loads(open(args.liteefg.format(game), 'rb').read())
        count = loads(open(args.count.format(game), 'rb').read())
        n = count['node_count']
        noregret_cuda = noregret_cuda['setup_time']
        liteefg = liteefg['setup_time']

        data['Game'].append(GAMES[game])
        data['# nodes'].append(scientific2(n))
        data[NOREGRET_CUDA].append(time(noregret_cuda))
        data[LITEEFG].append(time(liteefg))

    df = pd.DataFrame(data)

    df.to_latex(args.table)


if __name__ == '__main__':
    main()
