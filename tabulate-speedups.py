from argparse import ArgumentParser

from humanize import scientific
from pint import UnitRegistry
import pandas as pd

NOREGRET_CUDA = 'Ours (GPU)'
NOREGRET_FP = 'Ours (ST-CPU)'
UREG = UnitRegistry()


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('noregret_cuda')
    parser.add_argument('noregret_fp')
    parser.add_argument('table')

    return parser.parse_args()


def time(value):
    return f'{value * UREG.second:.1f~#P}'


def speedup(value):
    if value < 1:
        value = -1 / value

    return f'{value:.1f}' if -100 < value < 100 else scientific(value, 1)


def main():
    args = parse_args()
    df = pd.DataFrame()
    noregret_cuda = pd.read_csv(args.noregret_cuda)
    noregret_fp = pd.read_csv(args.noregret_fp)
    df['Operation'] = noregret_cuda['Operation']
    df[NOREGRET_CUDA] = noregret_cuda['Total'].map(time)
    df[NOREGRET_FP] = noregret_fp['Total'].map(time)
    df['Speedup'] = (noregret_fp['Total'] / noregret_cuda['Total']).map(speedup)

    df.to_latex(args.table)


if __name__ == '__main__':
    main()
