from pathlib import Path

from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np

from noregret.games import TwoPlayerZeroSumNormalFormGame
from noregret.regret_minimizers import (
    BlumMansour,
    EuclideanRegularization,
    MultiplicativeWeightsUpdate,
    OnlineGradientDescent,
    RegretMatching,
    RegretMatchingPlus,
)

FIGURE_PATH = Path(__file__).parent / 'figures' / 'rock-paper-scissors'
GAME_PATH = (
    Path(__file__).parent.parent.parent
    / 'games'
    / 'normal-form'
    / 'rock-paper-scissors.json'
)
NASH_EQUILIBRIUM = np.full(3, 1 / 3)
REGRET_MINIMIZERS = {
    'MWU': MultiplicativeWeightsUpdate(
        3,
        0.001,
        cumulative_utility=np.log(np.array([0.5, 0.25, 0.25])) / 0.001,
    ),
    'BM-MWU': BlumMansour(
        3,
        lambda _: MultiplicativeWeightsUpdate(
            3,
            0.001,
            cumulative_utility=np.log(np.array([0.5, 0.25, 0.25])) / 0.001,
        ),
    ),
    'ER': EuclideanRegularization(
        3,
        0.001,
        cumulative_utility=np.array([0.5, 0.25, 0.25]) / 0.001,
    ),
    'BM-ER': BlumMansour(
        3,
        lambda _: EuclideanRegularization(
            3,
            0.001,
            cumulative_utility=np.array([0.5, 0.25, 0.25]) / 0.001,
        ),
    ),
    'OGD': OnlineGradientDescent(
        3,
        0.001,
        previous_utility=np.array([0.5, 0.25, 0.25]) / 0.001,
    ),
    'BM-OGD': BlumMansour(
        3,
        lambda _: OnlineGradientDescent(
            3,
            0.001,
            previous_utility=np.array([0.5, 0.25, 0.25]) / 0.001,
        ),
    ),
    'RM': RegretMatching(3, cumulative_regrets=np.array([0.5, 0.25, 0.25])),
    'BM-RM': BlumMansour(
        3,
        lambda _: RegretMatching(
            3,
            cumulative_regrets=np.array([0.5, 0.25, 0.25]),
        ),
    ),
    'RM+': RegretMatchingPlus(
        3,
        cumulative_regrets_plus=np.array([0.5, 0.25, 0.25]),
    ),
    'BM-RM+': BlumMansour(
        3,
        lambda _: RegretMatchingPlus(
            3,
            cumulative_regrets_plus=np.array([0.5, 0.25, 0.25]),
        ),
    ),
}
ITERATION_COUNT = 10 ** 5


def main():
    with open(GAME_PATH) as file:
        game = TwoPlayerZeroSumNormalFormGame.load(file)

    for name, regret_minimizer in tqdm(REGRET_MINIMIZERS.items()):
        for iteration in trange(1, ITERATION_COUNT + 1, leave=False):
            strategy = regret_minimizer.next_strategy()
            utility = game.row_utility(strategy)

            regret_minimizer.observe_utility(utility)

        strategies = np.array(regret_minimizer.strategies)

        plt.clf()
        plt.plot(strategies[:, 0], strategies[:, 1])
        plt.plot(strategies[-1, 0], strategies[-1, 1], 'bo')
        plt.plot(*NASH_EQUILIBRIUM[:2], 'ro')
        plt.xlabel('Probability of action 1')
        plt.ylabel('Probability of action 2')
        plt.title(f'{name} in self-play')
        plt.savefig(FIGURE_PATH / f'{name}.pdf')


if __name__ == '__main__':
    main()
