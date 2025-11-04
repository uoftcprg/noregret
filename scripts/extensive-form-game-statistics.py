from sys import argv, stdout
from warnings import warn

import pandas as pd

from noregret.games import (
    ExtensiveFormGame,
    TwoPlayerExtensiveFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
)

GAME_TYPES = (
    TwoPlayerZeroSumExtensiveFormGame,
    TwoPlayerExtensiveFormGame,
    ExtensiveFormGame,
)
GAME_NAMES = argv[1::2]
GAME_PATHS = argv[2::2]


def main():
    data = {'Game': [], 'Type': [], 'Decision points': [], 'Sequences': []}

    for game_name, game_path in zip(GAME_NAMES, GAME_PATHS):
        with open(game_path) as file:
            content = file.read()

        game = None

        for game_type in GAME_TYPES:
            try:
                game = game_type.loads(content)
            except (KeyError, ValueError):
                pass
            else:
                break

        if game is None:
            warn(f'Unable to load {game_name}. Skipping...')

            continue

        if isinstance(game, TwoPlayerZeroSumExtensiveFormGame):
            game_type = '2p0s'
        elif isinstance(game, TwoPlayerExtensiveFormGame):
            game_type = '2p'
        else:
            game_type = None

        decision_point_count = 0
        sequence_count = 0

        for tfsdp in game.tree_form_sequential_decision_processes:
            decision_point_count += len(tfsdp.decision_points)
            sequence_count += len(tfsdp.sequences)

        data['Game'].append(game_name)
        data['Type'].append(game_type)
        data['Decision points'].append(decision_point_count)
        data['Sequences'].append(sequence_count)

    df = pd.DataFrame(data)

    df.to_csv(stdout)


if __name__ == '__main__':
    main()
