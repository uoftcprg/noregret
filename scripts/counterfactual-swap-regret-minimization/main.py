from dataclasses import dataclass, KW_ONLY
from functools import partial
from math import inf
from pathlib import Path
from sys import stdout
from typing import Any

from jsonlines import Writer
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from noregret.games import (
    TwoPlayerExtensiveFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
)
from noregret.regret_minimizers import (
    BlumMansour,
    CounterfactualRegretMinimizationPlus,
    RegretMatchingPlus,
)


@dataclass
class Variant:
    regret_minimizer_factory: Any
    _: KW_ONLY
    is_predictive: Any = False
    is_swap: Any = False


GAMES = {
    # 2p0s games

    'Kuhn poker': (
        TwoPlayerZeroSumExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
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
            / 'open-spiel'
            / 'leduc-poker.json'
        ),
        Path(__file__).parent / 'figures' / 'leduc-poker',
    ),
    'liar\'s dice': (
        TwoPlayerZeroSumExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / 'liars-dice.json'
        ),
        Path(__file__).parent / 'figures' / 'liars-dice',
    ),

    # 2p games

    'first sealed auction': (
        TwoPlayerExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / 'first-sealed-auction.json'
        ),
        Path(__file__).parent / 'figures' / 'first-sealed-auction',
    ),
    'sheriff': (
        TwoPlayerExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / 'sheriff.json'
        ),
        Path(__file__).parent / 'figures' / 'sheriff',
    ),
    'tiny bridge (2p)': (
        TwoPlayerExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / '2p-tiny-bridge.json'
        ),
        Path(__file__).parent / 'figures' / '2p-tiny-bridge',
    ),
    'tiny hanabi': (
        TwoPlayerExtensiveFormGame,
        (
            Path(__file__).parent.parent.parent
            / 'games'
            / 'extensive-form'
            / 'open-spiel'
            / 'tiny-hanabi.json'
        ),
        Path(__file__).parent / 'figures' / 'tiny-hanabi',
    ),
}
GAME_TYPES = {
    TwoPlayerZeroSumExtensiveFormGame: '2p0s',
    TwoPlayerExtensiveFormGame: '2p',
}
VARIANTS = {
    'CFR+': Variant(CounterfactualRegretMinimizationPlus),
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
    'PCFR+': Variant(
        partial(CounterfactualRegretMinimizationPlus, gamma=2),
        is_predictive=True,
    ),
    'BM-PCFR+': Variant(
        partial(
            CounterfactualRegretMinimizationPlus,
            regret_minimizer_factory=partial(
                BlumMansour,
                regret_minimizer_factory=RegretMatchingPlus,
            ),
            gamma=2,
        ),
        is_predictive=True,
        is_swap=True,
    ),
    'PCFR+ (γ=∞)': Variant(
        partial(CounterfactualRegretMinimizationPlus, gamma=inf),
        is_predictive=True,
    ),
    'BM-PCFR+ (γ=∞)': Variant(
        partial(
            CounterfactualRegretMinimizationPlus,
            regret_minimizer_factory=partial(
                BlumMansour,
                regret_minimizer_factory=RegretMatchingPlus,
            ),
            gamma=inf,
        ),
        is_predictive=True,
        is_swap=True,
    ),
}
ITERATION_COUNT = 10 ** 3


def main():
    average_strategies = []

    for game_name, (game_type, game_path, figure_path) in tqdm(GAMES.items()):
        with open(game_path) as file:
            game = game_type.load(file)

        variant_names = []
        regret_notions = []
        iterations = []
        nash_gaps = []
        cce_gaps = []
        values = []

        for variant_name, variant in tqdm(VARIANTS.items(), leave=False):
            row_cfr = variant.regret_minimizer_factory(
                game.row_tree_form_sequential_decision_process,
            )
            column_cfr = variant.regret_minimizer_factory(
                game.column_tree_form_sequential_decision_process,
            )

            for iteration in trange(1, ITERATION_COUNT + 1, leave=False):
                row_strategy = row_cfr.next_strategy(variant.is_predictive)

                if iteration > 1:
                    column_utility = game.column_utility(row_strategy)

                    column_cfr.observe_utility(column_utility)

                column_strategy = column_cfr.next_strategy(
                    variant.is_predictive,
                )
                row_utility = game.row_utility(column_strategy)

                row_cfr.observe_utility(row_utility)

                variant_names.append(variant_name)
                regret_notions.append(
                    'Swap' if variant.is_swap else 'External',
                )
                iterations.append(iteration)
                nash_gaps.append(
                    game.nash_gap(
                        row_cfr.average_strategy,
                        column_cfr.average_strategy,
                    ),
                )
                cce_gaps.append(
                    game.cce_gap(
                        row_cfr.strategies,
                        column_cfr.strategies,
                    ),
                )
                values.append(
                    game.values(
                        row_cfr.average_strategy,
                        column_cfr.average_strategy,
                    ),
                )

            average_strategies.append(
                {
                    'game': {
                        'name': game_name,
                        'type': GAME_TYPES.get(game_type),
                    },
                    'variant': {
                        'name': variant_name,
                        'is_predictive': variant.is_predictive,
                        'is_swap': variant.is_swap,
                    },
                    'average_strategies': {
                        'row': row_cfr.average_strategy.tolist(),
                        'column': column_cfr.average_strategy.tolist(),
                    },
                },
            )

        values = np.array(values)
        data = {
            'Variant': variant_names,
            'Regret notion': regret_notions,
            'Iteration': iterations,
            'Nash gap': nash_gaps,
            'CCE gap': cce_gaps,
            'Value': values[:, 0],
            'Row value': values[:, 0],
            'Column value': values[:, 1],
        }
        df = pd.DataFrame(data)

        plt.clf()
        sns.lineplot(
            df,
            x='Iteration',
            y='Nash gap',
            hue='Variant',
            style='Regret notion',
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Nash gap of {game_name} in self-play')
        plt.savefig(figure_path / 'nash-gap.pdf')

        plt.clf()
        sns.lineplot(
            df,
            x='Iteration',
            y='CCE gap',
            hue='Variant',
            style='Regret notion',
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'CCE gap of {game_name} in self-play')
        plt.savefig(figure_path / 'cce-gap.pdf')

        if isinstance(game, TwoPlayerZeroSumExtensiveFormGame):
            plt.clf()
            sns.lineplot(
                df,
                x='Iteration',
                y='Value',
                hue='Variant',
                style='Regret notion',
            )
            plt.title(f'Value of {game_name} in self-play')
            plt.savefig(figure_path / 'value.pdf')
        elif isinstance(game, TwoPlayerExtensiveFormGame):
            plt.clf()
            sns.lineplot(
                df,
                x='Iteration',
                y='Row value',
                hue='Variant',
                style='Regret notion',
            )
            plt.title(f'Row value of {game_name} in self-play')
            plt.savefig(figure_path / 'row-value.pdf')

            plt.clf()
            sns.lineplot(
                df,
                x='Iteration',
                y='Column value',
                hue='Variant',
                style='Regret notion',
            )
            plt.title(f'Column value of {game_name} in self-play')
            plt.savefig(figure_path / 'column-value.pdf')

    with Writer(stdout) as writer:
        writer.write_all(average_strategies)


if __name__ == '__main__':
    main()
