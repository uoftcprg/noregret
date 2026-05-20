"""Replicate Leme, Piliouras, and Schneider (NeurIPS, 2024)."""
from functools import partial

import matplotlib.pyplot as plt
import noregret as nr

KERNEL = nr.FloatingPointKernel()
GAME = nr.RockPaperScissorsPlus(KERNEL)
R_type = partial(nr.MWU, learning_rate=1e-3)


def main():
    RM = R_type(KERNEL, GAME.row_dimension, is_time_symmetric=False)
    BM_RM = nr.BM(KERNEL, GAME.row_dimension, R_type, is_time_symmetric=False)

    nr.symmetric_regret_minimization(GAME, RM, iteration_count=100000)
    nr.symmetric_regret_minimization(GAME, BM_RM, iteration_count=100000)
    x, _ = nr.linear_programming(GAME)

    strategies = KERNEL.numpy.array(RM.strategies)

    plt.clf()
    plt.plot(strategies[:, 0], strategies[:, 1])
    plt.plot(strategies[-1, 0], strategies[-1, 1], 'bo')
    plt.plot(*x[:2], 'ro')
    plt.xlabel('Probability of action 1')
    plt.ylabel('Probability of action 2')
    plt.title('No-external regret dynamics')
    plt.show()

    strategies = KERNEL.numpy.array(BM_RM.strategies)

    plt.clf()
    plt.plot(strategies[:, 0], strategies[:, 1])
    plt.plot(strategies[-1, 0], strategies[-1, 1], 'bo')
    plt.plot(*x[:2], 'ro')
    plt.xlabel('Probability of action 1')
    plt.ylabel('Probability of action 2')
    plt.title('No-swap regret dynamics')
    plt.show()


if __name__ == '__main__':
    main()
