"""Replicate Cai et al. (ICLR, 2025)."""
from math import inf

from tqdm import tqdm
import matplotlib.pyplot as plt
import noregret as nr
import pandas as pd
import seaborn as sns

KER = nr.FPKer()
np = KER.numpy
dtype = KER.data_type
A = np.array([[3, 0, -3], [0, 3, -4], [0, 0, 1]], dtype)
GAME = nr.matrix_game(KER, A)
PARAMETERS = {
    'PRM': (nr.RegretMatching, False),
    'PRM w/alt.': (nr.RegretMatching, True),
    'PRM+': (nr.RegretMatchingPlus, False),
    'PRM+ w/alt.': (nr.RegretMatchingPlus, True),
}
ITERATION_COUNT = 100000


def main():
    iterations = []
    exploitabilities = []
    variants = []

    for variant, (R_type, alt) in tqdm(PARAMETERS.items(), leave=False):
        R_row = R_type(KER, GAME.row_dimension, gamma=inf)
        R_col = R_type(KER, GAME.column_dimension, gamma=inf)

        def update():
            t = R_row.iteration_count
            x_bar = R_row.average_strategy
            y_bar = R_col.average_strategy
            epsilon = GAME.exploitability(x_bar, y_bar)

            iterations.append(t)
            exploitabilities.append(epsilon)
            variants.append(variant)

        nr.rm(
            GAME,
            R_row,
            R_col,
            alternation=alt,
            prediction=True,
            iteration_count=ITERATION_COUNT,
            update=update,
            progress_bar={'leave': False},
        )

    data = {
        'Iteration': iterations,
        'Exploitability': exploitabilities,
        'Variant': variants,
    }
    df = pd.DataFrame(data)

    plt.clf()
    sns.lineplot(df, x='Iteration', y='Exploitability', hue='Variant')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Exploitability')
    plt.show()


if __name__ == '__main__':
    main()
