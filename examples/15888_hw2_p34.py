"""Complete Problem 3.4 of Homework 2 from CMU graduate course 15-888:
Computational Game Solving.
"""
from functools import partial
from math import inf

from tqdm import tqdm
import matplotlib.pyplot as plt
import noregret as nr
import pandas as pd
import seaborn as sns

KERNEL = nr.FloatingPointKernel()
GAMES = {
    'Rock paper superscissors': nr.to_efg(nr.RockPaperSuperscissors(KERNEL)),
    'Kuhn poker': nr.from_open_spiel(KERNEL, 'kuhn_poker'),
    'Leduc poker': nr.from_open_spiel(KERNEL, 'leduc_poker'),
}
PARAMETERS = {
    'CFR': (nr.CFR, False, False),
    'CFR+': (nr.CFR_plus, True, False),
    'DCFR': (nr.DCFR, True, False),
    'PCFR+': (partial(nr.CFR_plus, gamma=2), True, True),
    'PCFR+*': (partial(nr.CFR_plus, gamma=inf), True, True),
}


def main():
    for name, game in tqdm(GAMES.items()):
        iterations = []
        exploitabilities = []
        expected_utilities = []
        variants = []

        for variant, (R_type, alt, pred) in tqdm(
                PARAMETERS.items(),
                leave=False,
        ):
            R_row = R_type(KERNEL, game.row_sequence_form_polytope)
            R_col = R_type(KERNEL, game.column_sequence_form_polytope)

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

            nr.regret_minimization(
                game,
                R_row,
                R_col,
                alternation=alt,
                prediction=pred,
                update=update,
                progress_bar={'leave': False},
            )

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
