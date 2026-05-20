"""Module or regret minimization."""
from collections.abc import Iterable, Mapping
from itertools import count

from tqdm import tqdm
import numpy as np


def regret_minimization(
        game,
        *regret_minimizers,
        alternation=False,
        prediction=False,
        iteration_count=1000,
        target_exploitability=None,
        checkpoints=(),
        update=None,
        progress_bar=True,
):
    """Solve a game using regret minimization.

    :param game: Game.
    :param regret_minimizers: Regret minimizers for players.
    :param alternation: Whether to alternate, defaults to ``True''.
    :param prediction: Whether to use optimism, defaults to ``False''.
    :param iteration_count: Number of iterations, defaults to ``1000''.
    :param target_exploitability: Optional target exploitability.
    :param checkpoints: Checkpoints.
    :param update: Update.
    :param progress_bar: Whether to show a progress bar.
    :return: Average strategy profile.
    """

    def average_strategy_profile():
        average_strategy_profile = []

        for R in regret_minimizers:
            average_strategy_profile.append(R.average_strategy)

        return tuple(average_strategy_profile)

    def exploitability():
        return game.exploitability(*average_strategy_profile())

    if iteration_count is None or np.isposinf(iteration_count):
        iterations = count()
    else:
        iterations = range(iteration_count)

    if progress_bar is True:
        iterations = tqdm(iterations)
    elif isinstance(progress_bar, Mapping):
        iterations = tqdm(iterations, **progress_bar)
    elif isinstance(progress_bar, Iterable):
        iterations = tqdm(iterations, *progress_bar)

    s = []

    for R in regret_minimizers:
        s.append(R.output(prediction))

    for i in iterations:
        if alternation:
            for j, R in enumerate(regret_minimizers):
                R.observe(game.utility(j, *s[:j], *s[j + 1:]))

                s[j] = R.output(prediction)
        else:
            U = game.utilities(*s)

            for j, (R, u) in enumerate(zip(regret_minimizers, U)):
                R.observe(u)

                s[j] = R.output(prediction)

        if not checkpoints or i in checkpoints:
            if update is not None:
                update()

            if (
                    target_exploitability is not None
                    and exploitability() < target_exploitability
            ):
                break

    return average_strategy_profile()


def symmetric_regret_minimization(
        game,
        regret_minimizer,
        prediction=False,
        iteration_count=1000,
        target_exploitability=None,
        checkpoints=(),
        update=None,
        progress_bar=True,
):
    """Solve a symmetric game using regret minimization under symmetry.

    :param game: Symmetric game.
    :param regret_minimizer: Regret minimizer.
    :param prediction: Whether to use optimism, defaults to ``False''.
    :param iteration_count: Number of iterations, defaults to ``1000''.
    :param target_exploitability: Optional target exploitability.
    :param checkpoints: Checkpoints.
    :param update: Update.
    :param progress_bar: Whether to show a progress bar.
    :return: Average strategy profile.
    """
    if not game.is_symmetric():
        raise ValueError('game is asymmetric')

    R = regret_minimizer

    def average_strategy_profile():
        return [R.average_strategy] * game.player_count

    def exploitability():
        return game.exploitability(*average_strategy_profile())

    if iteration_count is None or np.isposinf(iteration_count):
        iterations = count()
    else:
        iterations = range(iteration_count)

    if progress_bar is True:
        iterations = tqdm(iterations)
    elif isinstance(progress_bar, Mapping):
        iterations = tqdm(iterations, **progress_bar)
    elif isinstance(progress_bar, Iterable):
        iterations = tqdm(iterations, *progress_bar)

    s_neg_1 = [R.output(prediction)] * (game.player_count - 1)

    for i in iterations:
        R.observe(game.utility(0, *s_neg_1))

        s_neg_1 = [R.output(prediction)] * (game.player_count - 1)

        if not checkpoints or i in checkpoints:
            if update is not None:
                update()

            if (
                    target_exploitability is not None
                    and exploitability() < target_exploitability
            ):
                break

    return average_strategy_profile()
