"""Solution to Problem 3.1 of Homework 1 from CMU SCS CSD graduate
course 15-888: Computational Game Solving.
"""
import noregret as nr

KER = nr.FPKer()
GAMES = {
    'Rock paper superscissors': nr.RockPaperSuperscissors(KER),
    'Kuhn poker': nr.to_efg(KER, nr.open_spiel_game(KER, 'kuhn_poker')),
    'Leduc poker': nr.to_efg(KER, nr.open_spiel_game(KER, 'leduc_poker')),
}


def main():
    for name, game in GAMES.items():
        x, y = nr.lp(game)
        v = game.expected_row_utility(x, y)

        print(f'{name}:', v)


if __name__ == '__main__':
    main()
