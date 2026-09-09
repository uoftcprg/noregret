from argparse import ArgumentParser

from humanize import scientific
from pint import UnitRegistry
import pandas as pd

NOREGRET_GPU = 'Ours (GPU)'
NOREGRET_CPU = 'Ours (CPU)'
UREG = UnitRegistry()


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('noregret_gpu')
    parser.add_argument('noregret_cpu')
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
    noregret_gpu = pd.read_csv(args.noregret_gpu)
    noregret_cpu = pd.read_csv(args.noregret_cpu)
    df['Operation'] = noregret_gpu['Operation']
    df[NOREGRET_GPU] = noregret_gpu['Total'].map(time)
    df[NOREGRET_CPU] = noregret_cpu['Total'].map(time)
    df['Speedup'] = (noregret_cpu['Total'] / noregret_gpu['Total']).map(speedup)

    df.to_latex(args.table)


if __name__ == '__main__':
    main()
