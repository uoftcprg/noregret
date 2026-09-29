from argparse import ArgumentParser
from pathlib import Path

import noregret as nr


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('game')
    parser.add_argument('data', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    ker = nr.FPKer()
    game_type = nr.import_object(args.game)
    game = nr.PokerKitGame(ker, game_type).to_extensive_form()

    with open(args.data, 'wb') as file:
        file.write(game.dumps())


if __name__ == '__main__':
    main()
