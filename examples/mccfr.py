"""Run MCCFR."""
from functools import partial

from tqdm import tqdm
import matplotlib.pyplot as plt
import noregret as nr
import numpy as np
import pandas as pd
import seaborn as sns

KER = nr.FPKer()
GAME = nr.OpenSpielGame(KER, 'liars_dice')
R_types = {
    'External sampling': nr.MCCFR,
    'Outcome sampling': partial(
        nr.MCCFR,
        reference_strategy_profile=nr.UniformStrategyProfile(KER, GAME),
    ),
}
SAMPLE_COUNT = 100000
CHECKPOINTS = set((np.logspace(1, 5).round() - 1).astype(int).tolist())
SEED = 42


def main():
    for variant, R_type in tqdm(R_types.items()):
        R = R_type(KER, GAME)
        node_visit_counts = []
        exploitabilities = []

        def update():
            n = R.node_visit_count
            sigma = R.average_action_probabilities
            epsilon = GAME.exploitability(sigma)

            node_visit_counts.append(n)
            exploitabilities.append(epsilon)

        np.random.seed(SEED)
        nr.stochastic_rm(
            GAME,
            R,
            alternation=True,
            sample_count=SAMPLE_COUNT,
            checkpoints=CHECKPOINTS,
            update=update,
            progress_bar={'leave': False},
        )

        data = {
            '# node visits': node_visit_counts,
            'Exploitability': exploitabilities,
        }
        df = pd.DataFrame(data)

        plt.clf()
        sns.lineplot(df, x='# node visits', y='Exploitability')
        plt.yscale('log')
        plt.title(variant)
        plt.show()


if __name__ == '__main__':
    main()
