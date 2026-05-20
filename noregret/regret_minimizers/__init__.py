"""Module for regret minimizers."""
from noregret.regret_minimizers.regret_minimizers import (
    RegretMinimizer,
    SwapRegretMinimizer,
)
from noregret.regret_minimizers.probability_simplices import (
    BlumMansour,
    DiscountedRegretMatching,
    DiscountedRegretMinimizer,
    EuclideanRegularization,
    FollowTheRegularizedLeader,
    MirrorDescent,
    MultiplicativeWeightsUpdate,
    OnlineGradientDescent,
    ProbabilitySimplexRegretMinimizer,
    ProbabilitySimplexSwapRegretMinimizer,
    RegretMatching,
    RegretMatchingPlus,
)
from noregret.regret_minimizers.sequence_form_polytopes import (
    CounterfactualRegretMinimization,
    CounterfactualRegretMinimizationPlus,
    DiscountedCounterfactualRegretMinimization,
    SequenceFormPolytopeRegretMinimizer,
)

__all__ = (
    'BlumMansour',
    'CounterfactualRegretMinimization',
    'CounterfactualRegretMinimizationPlus',
    'DiscountedCounterfactualRegretMinimization',
    'DiscountedRegretMatching',
    'DiscountedRegretMinimizer',
    'EuclideanRegularization',
    'FollowTheRegularizedLeader',
    'MirrorDescent',
    'MultiplicativeWeightsUpdate',
    'OnlineGradientDescent',
    'ProbabilitySimplexRegretMinimizer',
    'ProbabilitySimplexSwapRegretMinimizer',
    'RegretMatching',
    'RegretMatchingPlus',
    'RegretMinimizer',
    'SequenceFormPolytopeRegretMinimizer',
    'SwapRegretMinimizer',
)
