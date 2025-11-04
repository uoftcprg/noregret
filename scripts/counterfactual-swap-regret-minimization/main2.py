from pathlib import Path
from sys import stdin

from jsonlines import Reader
import numpy as np
import pandas as pd

from noregret.games import (
    TwoPlayerExtensiveFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
)


GAMES = {
    # 2p0s games

    'Kuhn poker': (
        TwoPlayerZeroSumExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / 'kuhn-poker.json'
        ),
        Path(__file__).parent / 'data' / 'kuhn-poker',
    ),
    'Leduc poker': (
        TwoPlayerZeroSumExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / 'leduc-poker.json'
        ),
        Path(__file__).parent / 'data' / 'leduc-poker',
    ),
    'liar\'s dice': (
        TwoPlayerZeroSumExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / 'liars-dice.json'
        ),
        Path(__file__).parent / 'data' / 'liars-dice',
    ),

    # 2p games

    'first sealed auction': (
        TwoPlayerExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / 'first-sealed-auction.json'
        ),
        Path(__file__).parent / 'data' / 'first-sealed-auction',
    ),
    'sheriff': (
        TwoPlayerExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / 'sheriff.json'
        ),
        Path(__file__).parent / 'data' / 'sheriff',
    ),
    'tiny bridge (2p)': (
        TwoPlayerExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / '2p-tiny-bridge.json'
        ),
        Path(__file__).parent / 'data' / '2p-tiny-bridge',
    ),
    'tiny hanabi': (
        TwoPlayerExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / 'tiny-hanabi.json'
        ),
        Path(__file__).parent / 'data' / 'tiny-hanabi',
    ),
}
REGRET_MINIMIZERS = (
    'CFR+',
    'BM-CFR+',
    'PCFR+',
    'BM-PCFR+',
    'PCFR+ (γ=∞)',
    'BM-PCFR+ (γ=∞)',
)


def main():
    games = {}
    result_paths = {}

    for game_name, (game_type, game_path, result_path) in GAMES.items():
        with open(game_path) as file:
            games[game_name] = game_type.load(file)
            result_paths[game_name] = result_path

    with Reader(stdin) as reader:
        lines = list(reader)

    row_strategies = {}
    column_strategies = {}

    for line in lines:
        key = line['game']['name'], line['variant']['name']
        average_strategies = line['average_strategies']
        row_strategies[key] = np.array(average_strategies['row'])
        column_strategies[key] = np.array(average_strategies['column'])

    for game_name in GAMES:
        data = {
            'regret_minimizer': [],
            **{name: [] for name in REGRET_MINIMIZERS},
        }
        game = games[game_name]

        for hero in REGRET_MINIMIZERS:
            data['regret_minimizer'].append(hero)

            hero_row_strategy = row_strategies[game_name, hero]
            hero_column_strategy = column_strategies[game_name, hero]

            for villain in REGRET_MINIMIZERS:
                villain_row_strategy = row_strategies[game_name, villain]
                villain_column_strategy = column_strategies[game_name, villain]
                hero_row_value = game.row_value(
                    hero_row_strategy,
                    villain_column_strategy,
                )
                hero_column_value = game.column_value(
                    villain_row_strategy,
                    hero_column_strategy,
                )
                hero_value = (hero_row_value + hero_column_value) / 2

                data[villain].append(hero_value)

        df = pd.DataFrame(data)
        result_path = result_paths[game_name]

        df.to_csv(result_path / 'tournament.csv', index=False)


if __name__ == '__main__':
    main()
