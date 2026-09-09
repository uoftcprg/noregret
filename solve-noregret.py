from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import noregret as nr
import pandas as pd

from utilities import MethodTimer, TimedCFR


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('game')
    parser.add_argument('ker_path')
    parser.add_argument('iteration_count', type=int)
    parser.add_argument('data', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    ker = nr.FPKer()
    game = nr.OpenSpielGame(ker, args.game)
    ker = nr.import_object(args.ker_path)()
    game = nr.to_efg(ker, game)
    row_sfp = game.row_sequence_form_polytope
    R_row = TimedCFR(ker, row_sfp)
    col_sfp = game.column_sequence_form_polytope
    R_col = TimedCFR(ker, col_sfp)

    nr.rm(game, R_row, R_col, iteration_count=args.iteration_count)

    data = defaultdict(list)

    for timer in MethodTimer.instances:
        data['Operation'].append(timer.method.__qualname__)
        data['Total'].append(timer.total)
        data['Average'].append(timer.average)
        data['Standard error'].append(timer.standard_error)

    df = pd.DataFrame(data)

    df.to_csv(args.data)


if __name__ == '__main__':
    main()
