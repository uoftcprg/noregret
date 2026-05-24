"""Module for ``noregret''."""
from noregret.games import (
    AssuranceGame,
    BattleOfTheSexes,
    BlackBoxGame,
    Chicken,
    ExtensiveFormGame,
    Game,
    GiftExchangeGame,
    MatchingPennies,
    matrix_game,
    MultilinearGame,
    NormalFormGame,
    open_spiel_game,
    PrisonersDilemma,
    PureCoordination,
    RockPaperScissors,
    RockPaperScissorsPlus,
    RockPaperSuperscissors,
    Simulation,
    StagHunt,
    StrategyProfile,
    to_extensive_form_game,
    TwoPlayerExtensiveFormGame,
    TwoPlayerGame,
    TwoPlayerMultilinearGame,
    TwoPlayerNormalFormGame,
    TwoPlayerZeroSumExtensiveFormGame,
    TwoPlayerZeroSumGame,
    TwoPlayerZeroSumMultilinearGame,
    TwoPlayerZeroSumNormalFormGame,
    UniformStrategyProfile,
)
from noregret.kernels import (
    CUDAKernel,
    FloatingPointKernel,
    ImportedKernel,
    Kernel,
    Serializable,
)
from noregret.regret_minimizers import (
    BlumMansour,
    CounterfactualRegretMinimization,
    CounterfactualRegretMinimization2,
    CounterfactualRegretMinimizationPlus,
    DiscountedCounterfactualRegretMinimization,
    DiscountedRegretMatching,
    DiscountedRegretMinimizer,
    EuclideanRegularization,
    FollowTheRegularizedLeader,
    MirrorDescent,
    MonteCarloCounterfactualRegretMinimization,
    MultiplicativeWeightsUpdate,
    OnlineGradientDescent,
    ProbabilitySimplexRegretMinimizer,
    ProbabilitySimplexSwapRegretMinimizer,
    RegretMatching,
    RegretMatchingPlus,
    RegretMinimizer,
    SequenceFormPolytopeRegretMinimizer,
    StochasticRegretMinimizer,
    SwapRegretMinimizer,
)
from noregret.sequence_form_polytopes import SequenceFormPolytope
from noregret.solvers import (
    linear_programming,
    regret_minimization,
    stochastic_regret_minimization,
    symmetric_regret_minimization,
)
from noregret.utilities import import_object, tuple_or_none

BM = BlumMansour
"""Alias for :class:`noregret.BlumMansour`."""
CFR = CounterfactualRegretMinimization
"""Alias for :class:`noregret.CounterfactualRegretMinimization`."""
CFR2 = CounterfactualRegretMinimization2
"""Alias for :class:`noregret.CounterfactualRegretMinimization2`."""
CFR_plus = CounterfactualRegretMinimizationPlus
"""Alias for :class:`noregret.CounterfactualRegretMinimizationPlus`."""
CUDAKer = CUDAKernel
"""Alias for :class:`CUDAKernel`."""
DCFR = DiscountedCounterfactualRegretMinimization
"""Alias for
:class:`noregret.DiscountedCounterfactualRegretMinimization`.
"""
DRM = DiscountedRegretMatching
"""Alias for :class:`noregret.DiscountedRegretMatching`."""
EFG_2p0s = TwoPlayerZeroSumExtensiveFormGame
"""Alias for :class:`noregret.TwoPlayerZeroSumExtensiveFormGame`."""
EFG_2p = TwoPlayerExtensiveFormGame
"""Alias for :class:`noregret.TwoPlayerExtensiveFormGame`."""
EFG = ExtensiveFormGame
"""Alias for :class:`noregret.ExtensiveFormGame`."""
ER = EuclideanRegularization
"""Alias for :class:`noregret.EuclideanRegularization`."""
FPKer = FloatingPointKernel
"""Alias for :class:`FloatingPointKernel`."""
FTRL = FollowTheRegularizedLeader
"""Alias for :class:`noregret.FollowTheRegularizedLeader`."""
lp = linear_programming
"""Alias for :func:`noregret.linear_programming`."""
MCCFR = MonteCarloCounterfactualRegretMinimization
"""Alias for
:class:`noregret.MonteCarloCounterfactualRegretMinimization`.
"""
MD = MirrorDescent
"""Alias for :class:`noregret.MirrorDescent`."""
MWU = MultiplicativeWeightsUpdate
"""Alias for :class:`noregret.MultiplicativeWeightsUpdate`."""
NFG_2p0s = TwoPlayerZeroSumNormalFormGame
"""Alias for :class:`noregret.TwoPlayerZeroSumNormalFormGame`."""
NFG_2p = TwoPlayerNormalFormGame
"""Alias for :class:`noregret.TwoPlayerNormalFormGame`."""
NFG = NormalFormGame
"""Alias for :class:`noregret.NormalFormGame`."""
OGD = OnlineGradientDescent
"""Alias for :class:`noregret.OnlineGradientDescent`."""
RM_plus = RegretMatchingPlus
"""Alias for :class:`noregret.RegretMatchingPlus`."""
RM = RegretMatching
"""Alias for :class:`noregret.RegretMatching`."""
rm = regret_minimization
"""Alias for :func:`noregret.regret_minimization`."""
Sim = Simulation
"""Alias for :class:`noregret.Simulation`."""
stochastic_rm = stochastic_regret_minimization
"""Alias for :func:`noregret.stochastic_regret_minimization`."""
symmetric_rm = symmetric_regret_minimization
"""Alias for :func:`noregret.symmetric_regret_minimization`."""
to_efg = to_extensive_form_game
"""Alias for :func:`noregret.to_extensive_form_game`."""

__all__ = (
    'AssuranceGame',
    'BattleOfTheSexes',
    'BlackBoxGame',
    'BlumMansour',
    'BM',
    'CFR',
    'CFR_plus',
    'Chicken',
    'CounterfactualRegretMinimization',
    'CounterfactualRegretMinimization2',
    'CounterfactualRegretMinimizationPlus',
    'CUDAKer',
    'CUDAKernel',
    'DCFR',
    'DiscountedCounterfactualRegretMinimization',
    'DiscountedRegretMatching',
    'DiscountedRegretMinimizer',
    'DRM',
    'EFG',
    'EFG_2p',
    'EFG_2p0s',
    'ER',
    'EuclideanRegularization',
    'ExtensiveFormGame',
    'FloatingPointKernel',
    'FollowTheRegularizedLeader',
    'FPKer',
    'FTRL',
    'Game',
    'GiftExchangeGame',
    'ImportedKernel',
    'import_object',
    'Kernel',
    'linear_programming',
    'lp',
    'MatchingPennies',
    'matrix_game',
    'MCCFR',
    'MD',
    'MirrorDescent',
    'MonteCarloCounterfactualRegretMinimization',
    'MultilinearGame',
    'MultiplicativeWeightsUpdate',
    'MWU',
    'NFG',
    'NFG_2p',
    'NFG_2p0s',
    'NormalFormGame',
    'OGD',
    'OnlineGradientDescent',
    'open_spiel_game',
    'PrisonersDilemma',
    'ProbabilitySimplexRegretMinimizer',
    'ProbabilitySimplexSwapRegretMinimizer',
    'PureCoordination',
    'RegretMatching',
    'RegretMatchingPlus',
    'regret_minimization',
    'RegretMinimizer',
    'rm',
    'RM',
    'RM_plus',
    'RockPaperScissors',
    'RockPaperScissorsPlus',
    'RockPaperSuperscissors',
    'SequenceFormPolytope',
    'SequenceFormPolytopeRegretMinimizer',
    'Serializable',
    'Sim',
    'Simulation',
    'StagHunt',
    'stochastic_regret_minimization',
    'StochasticRegretMinimizer',
    'stochastic_rm',
    'StrategyProfile',
    'SwapRegretMinimizer',
    'symmetric_regret_minimization',
    'symmetric_rm',
    'to_efg',
    'to_extensive_form_game',
    'tuple_or_none',
    'TwoPlayerExtensiveFormGame',
    'TwoPlayerGame',
    'TwoPlayerMultilinearGame',
    'TwoPlayerNormalFormGame',
    'TwoPlayerZeroSumExtensiveFormGame',
    'TwoPlayerZeroSumGame',
    'TwoPlayerZeroSumMultilinearGame',
    'TwoPlayerZeroSumNormalFormGame',
    'UniformStrategyProfile',
)
