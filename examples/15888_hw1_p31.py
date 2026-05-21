"""Complete Problem 3.1 of Homework 1 from CMU graduate course 15-888:
Computational Game Solving.
"""
import noregret as nr

KERNEL = nr.FloatingPointKernel()
GAMES = {
    'Rock paper superscissors': nr.RockPaperSuperscissors(KERNEL),
    'Kuhn poker': nr.to_efg(KERNEL, nr.from_open_spiel('kuhn_poker')),
    'Leduc poker': nr.to_efg(KERNEL, nr.from_open_spiel('leduc_poker')),
}


def main():
    for name, game in GAMES.items():
        x, y = nr.linear_programming(game)
        v = game.expected_row_utility(x, y)

        print(f'{name}:', v)


if __name__ == '__main__':
    main()
