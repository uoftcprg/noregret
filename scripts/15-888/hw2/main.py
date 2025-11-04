from dataclasses import dataclass, KW_ONLY
from functools import partial
from math import inf
from pathlib import Path
from typing import Any

from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from noregret.games import TwoPlayerZeroSumExtensiveFormGame
from noregret.regret_minimizers import (
    CounterfactualRegretMinimization,
    CounterfactualRegretMinimizationPlus,
    DiscountedCounterfactualRegretMinimization,
)


@dataclass
class Variant:
    regret_minimizer_factory: Any
    _: KW_ONLY
    is_alternating: Any = False
    is_predictive: Any = False


GAMES = {
    'rock paper superscissors': (
        (
            Path(__file__).parent.parent.parent.parent
            / 'games'
            / 'extensive-form'
            / '15-888'
            / 'rock-paper-superscissors.json'
        ),
        Path(__file__).parent / 'figures' / 'rock-paper-superscissors',
    ),
    'Kuhn poker': (
        (
            Path(__file__).parent.parent.parent.parent
            / 'games'
            / 'extensive-form'
            / '15-888'
            / 'kuhn-poker.json'
        ),
        Path(__file__).parent / 'figures' / 'kuhn-poker',
    ),
    'Leduc poker': (
        (
            Path(__file__).parent.parent.parent.parent
            / 'games'
            / 'extensive-form'
            / '15-888'
            / 'leduc-poker.json'
        ),
        Path(__file__).parent / 'figures' / 'leduc-poker',
    ),
}
VARIANTS = {
    'CFR': Variant(CounterfactualRegretMinimization),
    'CFR+, w/alt, γ=1': Variant(
        CounterfactualRegretMinimizationPlus,
        is_alternating=True,
    ),
    'DCFR, w/alt, α=1.5,β=0,γ=2': Variant(
        DiscountedCounterfactualRegretMinimization,
        is_alternating=True,
    ),
    'PCFR+, w/alt, γ=2': Variant(
        partial(CounterfactualRegretMinimizationPlus, gamma=2),
        is_alternating=True,
        is_predictive=True,
    ),
    'PCFR+, w/alt, γ=∞': Variant(
        partial(CounterfactualRegretMinimizationPlus, gamma=inf),
        is_alternating=True,
        is_predictive=True,
    ),
}
ITERATION_COUNT = 10 ** 3


def main():
    for game_name, (game_path, figure_path) in tqdm(GAMES.items()):
        with open(game_path) as file:
            game = TwoPlayerZeroSumExtensiveFormGame.load(file)

        variant_names = []
        iterations = []
        nash_gaps = []
        values = []

        for variant_name, variant in tqdm(VARIANTS.items(), leave=False):
            row_cfr = variant.regret_minimizer_factory(
                game.row_tree_form_sequential_decision_process,
            )
            column_cfr = variant.regret_minimizer_factory(
                game.column_tree_form_sequential_decision_process,
            )

            for iteration in trange(1, ITERATION_COUNT + 1, leave=False):
                if variant.is_alternating:
                    row_strategy = row_cfr.next_strategy(variant.is_predictive)

                    if iteration > 1:
                        column_utility = game.column_utility(row_strategy)

                        column_cfr.observe_utility(column_utility)

                    column_strategy = column_cfr.next_strategy(
                        variant.is_predictive,
                    )
                    row_utility = game.row_utility(column_strategy)

                    row_cfr.observe_utility(row_utility)
                else:
                    row_strategy = row_cfr.next_strategy(variant.is_predictive)
                    column_strategy = column_cfr.next_strategy(
                        variant.is_predictive,
                    )
                    row_utility = game.row_utility(column_strategy)
                    column_utility = game.column_utility(row_strategy)

                    row_cfr.observe_utility(row_utility)
                    column_cfr.observe_utility(column_utility)

                variant_names.append(variant_name)
                iterations.append(iteration)
                nash_gaps.append(
                    game.nash_gap(
                        row_cfr.average_strategy,
                        column_cfr.average_strategy,
                    ),
                )
                values.append(
                    game.row_value(
                        row_cfr.average_strategy,
                        column_cfr.average_strategy,
                    ),
                )

        data = {
            'Variant': variant_names,
            'Iteration': iterations,
            'Nash gap': nash_gaps,
            'Value': values,
        }
        df = pd.DataFrame(data)

        plt.clf()
        sns.lineplot(df, x='Iteration', y='Nash gap', hue='Variant')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Nash gap of {game_name} in self-play')
        plt.savefig(figure_path / 'nash-gap.pdf')

        plt.clf()
        sns.lineplot(df, x='Iteration', y='Value', hue='Variant')
        plt.xscale('log')
        plt.title(f'Value of {game_name} in self-play')
        plt.savefig(figure_path / 'value.pdf')


if __name__ == '__main__':
    main()
