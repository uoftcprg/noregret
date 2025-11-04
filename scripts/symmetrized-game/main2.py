from dataclasses import dataclass, KW_ONLY
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from noregret.games import TwoPlayerZeroSumExtensiveFormGame
from noregret.regret_minimizers import (
    BlumMansour,
    CounterfactualRegretMinimizationPlus,
    RegretMatchingPlus,
)


@dataclass
class Variant:
    regret_minimizer_factory: Any
    _: KW_ONLY
    is_swap: Any = False


GAMES = {
    'Kuhn poker': (
        TwoPlayerZeroSumExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / '15-888'
            / 'kuhn-poker.json'
        ),
        Path(__file__).parent / 'figures' / 'kuhn-poker',
    ),
    'Leduc poker': (
        TwoPlayerZeroSumExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / '15-888'
            / 'leduc-poker.json'
        ),
        Path(__file__).parent / 'figures' / 'leduc-poker',
    ),
}
VARIANTS = {
    'CFR+': Variant(partial(CounterfactualRegretMinimizationPlus)),
    'BM-CFR+': Variant(
        partial(
            CounterfactualRegretMinimizationPlus,
            regret_minimizer_factory=partial(
                BlumMansour,
                regret_minimizer_factory=RegretMatchingPlus,
            ),
        ),
        is_swap=True,
    ),
}
ITERATION_COUNT = 10 ** 4


def main():
    for game_name, (game_type, game_path, figure_path) in tqdm(GAMES.items()):
        with open(game_path) as file:
            game = game_type.load(file)

        variant_names = []
        regret_notions = []
        iterations = []
        nash_gaps = []
        convergences = []

        for variant_name, variant in tqdm(VARIANTS.items(), leave=False):
            row_cfr = variant.regret_minimizer_factory(
                game.row_tree_form_sequential_decision_process,
            )
            column_cfr = variant.regret_minimizer_factory(
                game.column_tree_form_sequential_decision_process,
            )

            for iteration in trange(1, ITERATION_COUNT + 1, leave=False):
                row_strategy = row_cfr.next_strategy()
                column_strategy = column_cfr.next_strategy()
                row_utility = game.row_utility(column_strategy)
                column_utility = game.column_utility(row_strategy)

                row_cfr.observe_utility(row_utility)
                column_cfr.observe_utility(column_utility)

                variant_names.extend([variant_name] * 2)
                regret_notions.extend(
                    ['Swap' if variant.is_swap else 'External'] * 2,
                )
                iterations.extend([iteration] * 2)
                nash_gaps.extend(
                    [
                        game.nash_gap(row_strategy, column_strategy),
                        game.nash_gap(
                            row_cfr.average_strategy,
                            column_cfr.average_strategy,
                        ),
                    ],
                )
                convergences.extend(['Last-iterate', 'Time-averaged'])

        data = {
            'Variant': variant_names,
            'Regret notion': regret_notions,
            'Iteration': iterations,
            'Nash gap': nash_gaps,
            'Convergences': convergences,
        }
        df = pd.DataFrame(data)

        plt.clf()
        sns.lineplot(
            df[df['Convergences'] == 'Last-iterate'],
            x='Iteration',
            y='Nash gap',
            hue='Variant',
            style='Regret notion',
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Nash gap of {game_name} in self-play (last-iterate)')
        plt.savefig(figure_path / 'last-iterate-nash-gap.pdf')

        plt.clf()
        sns.lineplot(
            df[df['Convergences'] == 'Time-averaged'],
            x='Iteration',
            y='Nash gap',
            hue='Variant',
            style='Regret notion',
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Nash gap of {game_name} in self-play (time-averaged)')
        plt.savefig(figure_path / 'time-averaged-nash-gap.pdf')


if __name__ == '__main__':
    main()
