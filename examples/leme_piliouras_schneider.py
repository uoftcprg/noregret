"""Replicate Leme, Piliouras, and Schneider (NeurIPS, 2024)."""
from functools import partial

import matplotlib.pyplot as plt
import noregret as nr

KER = nr.FPKer()
GAME = nr.RockPaperScissorsPlus(KER)
R_type = partial(nr.MWU, learning_rate=1e-3)


def main():
    x, _ = nr.lp(GAME)
    RM = R_type(KER, GAME.row_dimension, is_time_symmetric=False)

    nr.symmetric_rm(GAME, RM, iteration_count=100000)

    strategies = KER.numpy.array(RM.strategies)

    plt.clf()
    plt.plot(strategies[:, 0], strategies[:, 1])
    plt.plot(strategies[-1, 0], strategies[-1, 1], 'bo')
    plt.plot(*x[:2], 'ro')
    plt.xlabel('Probability of action 1')
    plt.ylabel('Probability of action 2')
    plt.title('No-external regret dynamics')
    plt.show()

    BM_RM = nr.BM(KER, GAME.row_dimension, R_type, is_time_symmetric=False)

    nr.symmetric_rm(GAME, BM_RM, iteration_count=100000)

    strategies = KER.numpy.array(BM_RM.strategies)

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
