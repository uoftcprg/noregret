"""Module or regret minimization."""
from collections.abc import Iterable, Mapping
from itertools import count, repeat

from tqdm import tqdm


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

    np = game.kernel.numpy

    if len(regret_minimizers) != game.player_count:
        raise ValueError('inconsistent number of regret minimizers')

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

    sigma = []

    for R in regret_minimizers:
        sigma.append(R.output(prediction))

    for t in iterations:
        if alternation:
            for i, R in enumerate(regret_minimizers):
                R.observe(game.utility(i, *sigma[:i], *sigma[i + 1:]))

                sigma[i] = R.output(prediction)
        else:
            us = game.utilities(*sigma)

            for i, (R, u) in enumerate(zip(regret_minimizers, us)):
                R.observe(u)

                sigma[i] = R.output(prediction)

        if not checkpoints or t in checkpoints:
            if update is not None:
                status = update()
            else:
                status = False

            if (
                    status
                    or (
                        target_exploitability is not None
                        and exploitability() < target_exploitability
                    )
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

    def average_strategy_profile():
        return [regret_minimizer.average_strategy] * game.player_count

    def exploitability():
        return game.exploitability(*average_strategy_profile())

    np = game.kernel.numpy

    if not game.is_symmetric:
        raise ValueError('game is asymmetric')

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

    sigma_1 = regret_minimizer.output(prediction)

    for t in iterations:
        u = game.utility(0, *repeat(sigma_1, game.player_count - 1))

        regret_minimizer.observe(u)

        sigma_1 = regret_minimizer.output(prediction)

        if not checkpoints or t in checkpoints:
            if update is not None:
                status = update()
            else:
                status = False

            if (
                    status
                    or (
                        target_exploitability is not None
                        and exploitability() < target_exploitability
                    )
            ):
                break

    return average_strategy_profile()


def stochastic_regret_minimization(
        game,
        regret_minimizer,
        alternation=False,
        sample_count=1000000,
        checkpoints=(),
        update=None,
        progress_bar=True,
):
    """Solve a game using stochastic regret minimization.

    :param game: Game.
    :param regret_minimizer: Regret minimizer.
    :param alternation: Whether to alternate, defaults to ``True''.
    :param sample_count: Number of samples, defaults to ``1000000''.
    :param checkpoints: Checkpoints.
    :param update: Update.
    :param progress_bar: Whether to show a progress bar.
    :return: Average action probabilities.
    """
    np = game.kernel.numpy

    if sample_count is None or np.isposinf(sample_count):
        samples = count()
    else:
        samples = range(sample_count)

    if progress_bar is True:
        samples = tqdm(samples)
    elif isinstance(progress_bar, Mapping):
        samples = tqdm(samples, **progress_bar)
    elif isinstance(progress_bar, Iterable):
        samples = tqdm(samples, *progress_bar)

    for s in samples:
        if alternation:
            for i in range(game.player_count):
                regret_minimizer.observe(regret_minimizer.sample(i))
        else:
            uss = []

            for i in range(game.player_count):
                uss.append(regret_minimizer.sample(i))

            for us in uss:
                regret_minimizer.observe(us)

        if not checkpoints or s in checkpoints:
            if update is not None:
                status = update()
            else:
                status = False

            if status:
                break

    return regret_minimizer.average_action_probabilities
