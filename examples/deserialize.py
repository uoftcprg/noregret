"""Deserialize (and solve) games."""
from itertools import chain, repeat
from pathlib import Path

import noregret as nr


KER = nr.CUDAKer(data_type='float32')
OPEN_SPIEL_PATH = Path(__file__).parent / 'games' / 'open-spiel'
OPEN_SPIEL_GAMES = (
    'kuhn-poker',
    'leduc-poker',
    'liars-dice',
    'goofspiel-6',
    'goofspiel-7',
    'battleship-3x2-2-3',
    'battleship-3x2-22-3',
)
POKERKIT_PATH = Path(__file__).parent / 'games' / 'pokerkit'
POKERKIT_GAMES = (
    'kuhn-poker',
    'leduc-holdem',
    'royal-rhode-island-holdem',
)


def main():
    for path, name in chain(
            zip(repeat(OPEN_SPIEL_PATH), OPEN_SPIEL_GAMES),
            zip(repeat(POKERKIT_PATH), POKERKIT_GAMES),
    ):
        with open(path / f'{name}.json', 'rb') as file:
            game = nr.EFG_2p0s.loads(KER, file.read())

        R_row = nr.CFR_plus(KER, game.row_sequence_form_polytope, gamma=2)
        R_col = nr.CFR_plus(KER, game.column_sequence_form_polytope, gamma=2)
        x, y = nr.rm(
            game,
            R_row,
            R_col,
            alternation=True,
            prediction=True,
            progress_bar=False,
        )
        v = game.expected_row_utility(x, y)

        print(f'{name}:', v)


if __name__ == '__main__':
    main()
