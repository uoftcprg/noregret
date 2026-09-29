"""Serialize games."""
from pathlib import Path

from pokerkit import KuhnPoker, LeducHoldem, RoyalRhodeIslandHoldem
from tqdm import tqdm
import noregret as nr


KER = nr.FPKer()
OPEN_SPIEL_PATH = Path(__file__).parent / 'games' / 'open-spiel'
OPEN_SPIEL_GAMES = {
    'kuhn-poker': 'kuhn_poker',
    'leduc-poker': 'leduc_poker',
    'liars-dice': 'liars_dice',
    'goofspiel-6': (
        'turn_based_simultaneous_game('
        'game=goofspiel(imp_info=True,num_cards=6,points_order=descending))'
    ),
    'goofspiel-7': (
        'turn_based_simultaneous_game('
        'game=goofspiel(imp_info=True,num_cards=7,points_order=descending))'
    ),
    'battleship-3x2-2-3': (
        'battleship('
        'board_height=3,'
        'board_width=2,'
        'ship_sizes=[2],'
        'ship_values=[4],'
        'num_shots=3)'
    ),
    'battleship-3x2-22-3': (
        'battleship('
        'board_height=3,'
        'board_width=2,'
        'ship_sizes=[2;2],'
        'ship_values=[4;4],'
        'num_shots=3)'
    ),
}
POKERKIT_PATH = Path(__file__).parent / 'games' / 'pokerkit'
POKERKIT_GAME_TYPES = {
    'kuhn-poker': KuhnPoker,
    'leduc-holdem': LeducHoldem,
    'royal-rhode-island-holdem': RoyalRhodeIslandHoldem,
}


def main():
    for name, game in tqdm(OPEN_SPIEL_GAMES.items()):
        game = nr.OpenSpielGame(KER, game)
        game = nr.to_efg(KER, game)

        with open(OPEN_SPIEL_PATH / f'{name}.json', 'wb') as file:
            file.write(game.dumps())

    for name, game_type in tqdm(POKERKIT_GAME_TYPES.items()):
        game = nr.PokerKitGame(KER, game_type).to_extensive_form()

        with open(POKERKIT_PATH / f'{name}.json', 'wb') as file:
            file.write(game.dumps())


if __name__ == '__main__':
    main()
