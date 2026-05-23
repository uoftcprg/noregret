"""Run CFR with MWU as local regret minimizers."""
from functools import partial

from tqdm import tqdm
import matplotlib.pyplot as plt
import noregret as nr
import pandas as pd
import seaborn as sns

KER = nr.FPKer()
GAMES = {
    'Rock paper superscissors': nr.to_efg(KER, nr.RockPaperSuperscissors(KER)),
    'Kuhn poker': nr.to_efg(KER, nr.open_spiel_game(KER, 'kuhn_poker')),
    'Leduc poker': nr.to_efg(KER, nr.open_spiel_game(KER, 'leduc_poker')),
}
PARAMETERS = {
    'CFR': nr.CFR,
    'CFR-MWU': partial(
        nr.CFR2,
        regret_minimizer_type=partial(nr.MWU, learning_rate=10),
    ),
}
PROGRESS_BAR = {'leave': False}


def main():
    for name, game in tqdm(GAMES.items()):
        iterations = []
        exploitabilities = []
        expected_utilities = []
        variants = []

        for variant, R_type in tqdm(
                PARAMETERS.items(),
                leave=False,
        ):
            R_row = R_type(KER, game.row_sequence_form_polytope)
            R_col = R_type(KER, game.column_sequence_form_polytope)

            def update():
                t = R_row.iteration_count
                x_bar = R_row.average_strategy
                y_bar = R_col.average_strategy
                epsilon = game.exploitability(x_bar, y_bar)
                u = game.expected_row_utility(x_bar, y_bar)

                iterations.append(t)
                exploitabilities.append(epsilon)
                expected_utilities.append(u)
                variants.append(variant)

            nr.rm(game, R_row, R_col, update=update, progress_bar=PROGRESS_BAR)

        data = {
            'Iteration': iterations,
            'Exploitability': exploitabilities,
            'Expected utility': expected_utilities,
            'Variant': variants,
        }
        df = pd.DataFrame(data)

        plt.clf()
        sns.lineplot(df, x='Iteration', y='Exploitability', hue='Variant')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Exploitability in {name}')
        plt.show()

        plt.clf()
        sns.lineplot(df, x='Iteration', y='Expected utility', hue='Variant')
        plt.xscale('log')
        plt.title(f'Expected utility in {name}')
        plt.show()


if __name__ == '__main__':
    main()
