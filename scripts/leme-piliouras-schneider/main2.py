from itertools import chain, repeat
from pathlib import Path

from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from noregret.games import TwoPlayerZeroSumNormalFormGame
from noregret.regret_minimizers import (
    BlumMansour,
    EuclideanRegularization,
    MultiplicativeWeightsUpdate,
    OnlineGradientDescent,
    RegretMatching,
    RegretMatchingPlus,
)

FIGURE_PATH = Path(__file__).parent / 'figures' / 'matching-pennies'
GAME_PATH = (
    Path(__file__).parent.parent.parent
    / 'games'
    / 'normal-form'
    / 'matching-pennies.json'
)
NASH_EQUILIBRIUM = np.full(2, 1 / 2)
REGRET_MINIMIZER_FACTORIES = {
    'MWU': lambda: MultiplicativeWeightsUpdate(
        2,
        0.001,
        cumulative_utility=np.log(np.array([0.75, 0.25])) / 0.001,
    ),
    'BM-MWU': lambda: BlumMansour(
        2,
        lambda _: MultiplicativeWeightsUpdate(
            2,
            0.001,
            cumulative_utility=np.log(np.array([0.75, 0.25])) / 0.001,
        ),
    ),
    'ER': lambda: EuclideanRegularization(
        2,
        0.001,
        cumulative_utility=np.array([0.75, 0.25]) / 0.001,
    ),
    'BM-ER': lambda: BlumMansour(
        2,
        lambda _: EuclideanRegularization(
            2,
            0.001,
            cumulative_utility=np.array([0.75, 0.25]) / 0.001,
        ),
    ),
    'OGD': lambda: OnlineGradientDescent(
        2,
        0.001,
        previous_utility=np.array([0.75, 0.25]) / 0.001,
    ),
    'BM-OGD': lambda: BlumMansour(
        2,
        lambda _: OnlineGradientDescent(
            2,
            0.001,
            previous_utility=np.array([0.75, 0.25]) / 0.001,
        ),
    ),
    'RM': lambda: RegretMatching(2, cumulative_regrets=np.array([0.75, 0.25])),
    'BM-RM': lambda: BlumMansour(
        2,
        lambda _: RegretMatching(2, cumulative_regrets=np.array([0.75, 0.25])),
    ),
    'RM+': lambda: RegretMatchingPlus(
        2,
        cumulative_regrets_plus=np.array([0.75, 0.25]),
    ),
    'BM-RM+': lambda: BlumMansour(
        2,
        lambda _: RegretMatchingPlus(
            2,
            cumulative_regrets_plus=np.array([0.75, 0.25]),
        ),
    ),
}
ITERATION_COUNT = 10 ** 5


def main():
    with open(GAME_PATH) as file:
        game = TwoPlayerZeroSumNormalFormGame.load(file)

    for name, regret_minimizer_factory in tqdm(
            REGRET_MINIMIZER_FACTORIES.items(),
    ):
        row_regret_minimizer = regret_minimizer_factory()
        column_regret_minimizer = regret_minimizer_factory()
        iterations = []

        for iteration in trange(1, ITERATION_COUNT + 1, leave=False):
            row_strategy = row_regret_minimizer.next_strategy()
            column_strategy = column_regret_minimizer.next_strategy()
            row_utility = game.row_utility(column_strategy)
            column_utility = game.column_utility(row_strategy)

            row_regret_minimizer.observe_utility(row_utility)
            column_regret_minimizer.observe_utility(column_utility)
            iterations.append(iteration)

        row_strategies = np.array(row_regret_minimizer.strategies)
        column_strategies = np.array(column_regret_minimizer.strategies)
        data = {
            'Iteration': chain.from_iterable(repeat(iterations, 2)),
            'Probability of action 1': chain(
                row_strategies[:, 0],
                column_strategies[:, 0],
            ),
            'Player': chain(
                repeat('Row', ITERATION_COUNT),
                repeat('Column', ITERATION_COUNT),
            ),
        }
        df = pd.DataFrame(data)

        plt.clf()
        sns.lineplot(
            df,
            x='Iteration',
            y='Probability of action 1',
            hue='Player',
        )
        plt.plot(iterations, [NASH_EQUILIBRIUM[0]] * ITERATION_COUNT, 'r--')
        plt.xlabel('Iteration')
        plt.ylabel('Probability of action 1')
        plt.title(f'{name} in self-play')
        plt.savefig(FIGURE_PATH / f'{name}.pdf')


if __name__ == '__main__':
    main()
