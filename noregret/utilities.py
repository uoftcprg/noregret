from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import auto, IntEnum
from functools import partial
from importlib import import_module
from json import dump, dumps, load, loads
from math import inf
from typing import Any

from ordered_set import OrderedSet
import numpy as np
import numpy.linalg as LA


def euclidean_projection_on_probability_simplex(input_):
    """Euclidean projection of the input on a probability simplex.

    >>> euclidean_projection_on_probability_simplex(np.array([0.2, 0.5, 0.3]))
    array([0.2, 0.5, 0.3])
    >>> euclidean_projection_on_probability_simplex(np.array([0.2, -0.3, 2]))
    array([0., 0., 1.])
    >>> euclidean_projection_on_probability_simplex(np.array([5, 5, 5, 5]))
    array([0.25, 0.25, 0.25, 0.25])
    >>> euclidean_projection_on_probability_simplex(np.array([10, 0, 0]))
    array([1., 0., 0.])
    >>> euclidean_projection_on_probability_simplex(np.array([0.6]))
    array([1.])
    >>> euclidean_projection_on_probability_simplex(np.array([0, 0, 0, 0, 0]))
    array([0.2, 0.2, 0.2, 0.2, 0.2])

    :param input_: The input to be projected.
    :return: The projection output.
    """
    sorted_input = np.flip(np.sort(input_))
    cumsum_sorted_input = sorted_input.cumsum()
    indices = np.arange(1, input_.size + 1)
    conditions = (sorted_input + (1 - cumsum_sorted_input) / indices) > 0
    rho = np.where(conditions)[0].max() + 1
    lambda_ = (1 - cumsum_sorted_input[rho - 1]) / rho
    output = (input_ + lambda_).clip(0)

    return output


def stationary_distribution(stochastic_matrix):
    P = stochastic_matrix

    if not np.allclose(P.sum(1), 1):
        raise ValueError('matrix not stochastic')

    eigenvalues, eigenvectors = LA.eig(P.T)
    pi = eigenvectors[:, np.isclose(eigenvalues, 1)][:, 0]
    pi /= pi.sum()
    pi = pi.real

    assert np.allclose(pi @ P, pi)

    return pi


def import_string(dotted_path):
    """Import an object from a module.

    >>> import_string('math.inf')
    inf

    :param dotted_path: The dotted path of the object to import.
    :return: Imported object.
    """
    module_path, class_name = dotted_path.rsplit('.', 1)

    return getattr(import_module(module_path), class_name)


def split(values, counts):
    """Split concatenated values.

    >>> split([0, 1, 2, 3, 4, 5], [3, 0, 1, 2])
    [[0, 1, 2], [], [3], [4, 5]]

    :param values: Values to be split.
    :param counts: The size of the partitions.
    :return: The split values.
    """
    splits = []
    begin = 0

    for count in counts:
        end = begin + count

        splits.append(values[begin:end])

        begin = end

    return splits


class Serializable(ABC):
    @classmethod
    @abstractmethod
    def deserialize(cls, raw_data):
        pass

    @classmethod
    def load(cls, *args, **kwargs):
        return cls.deserialize(load(*args, **kwargs))

    @classmethod
    def loads(cls, *args, **kwargs):
        return cls.deserialize(loads(*args, **kwargs))

    @abstractmethod
    def serialize(self):
        pass

    def dump(self, *args, **kwargs):
        return dump(self.serialize(), *args, **kwargs)

    def dumps(self, *args, **kwargs):
        return dumps(self.serialize(), *args, **kwargs)


@dataclass
class TreeFormSequentialDecisionProcess(Serializable):
    class NodeType(IntEnum):
        DECISION_POINT = auto()
        OBSERVATION_POINT = auto()
        END_OF_THE_DECISION_PROCESS = auto()

    @classmethod
    def deserialize_all(cls, raw_data):
        return list(map(cls.deserialize, raw_data))

    @classmethod
    def deserialize(cls, raw_data):
        transitions = {}
        node_types = {}

        for raw_transition in raw_data:
            parent_edge = tuple(raw_transition['parent_edge'])
            node_id = raw_transition['node']['id']
            node_type = getattr(cls.NodeType, raw_transition['node']['type'])
            transitions[parent_edge] = node_id
            node_types[node_id] = node_type

        return cls(transitions, node_types)

    transitions: Any
    """Entries are assumed be topologically sorted."""
    node_types: Any
    nodes: Any = field(init=False, default_factory=OrderedSet)
    decision_points: Any = field(init=False, default_factory=OrderedSet)
    observation_points: Any = field(init=False, default_factory=OrderedSet)
    sequences: Any = field(init=False, default_factory=OrderedSet)
    indices: Any = field(init=False, default_factory=dict)
    parent_sequences: Any = field(init=False, default_factory=dict)
    actions: Any = field(
        init=False,
        default_factory=partial(defaultdict, OrderedSet),
    )
    signals: Any = field(
        init=False,
        default_factory=partial(defaultdict, OrderedSet),
    )

    def __post_init__(self):
        self.nodes.update(self.transitions.values())

        for parent_edge, p in self.transitions.items():
            match self.node_types[p]:
                case self.NodeType.DECISION_POINT:
                    self.decision_points.add(p)
                case self.NodeType.OBSERVATION_POINT:
                    self.observation_points.add(p)
                case self.NodeType.END_OF_THE_DECISION_PROCESS:
                    pass
                case _:
                    raise ValueError('unknown node type')

            if parent_edge:
                parent, event = parent_edge

                match self.node_types[parent]:
                    case self.NodeType.DECISION_POINT:
                        is_sequence = True
                        parent_sequence = parent_edge

                        self.actions[parent].add(event)
                    case self.NodeType.OBSERVATION_POINT:
                        is_sequence = False
                        parent_sequence = self.parent_sequences[parent]

                        self.signals[parent].add(event)
                    case self.NodeType.END_OF_THE_DECISION_PROCESS:
                        raise ValueError('parent is an end of the tfsdp')
                    case _:
                        raise ValueError('unknown parent node type')
            else:
                is_sequence = True
                parent_sequence = parent_edge

            if is_sequence:
                self.sequences.add(parent_edge)

                self.indices[parent_edge] = len(self.indices)

            self.parent_sequences[p] = parent_sequence

    def behavioral_uniform_strategy(self):
        strategy = {
            j: np.full(len(self.actions[j]), 1 / len(self.actions[j]))
            for j in self.decision_points
        }

        return strategy

    def behavioral_best_response(self, utility):
        strategy = {}
        V = defaultdict(int)

        for p in reversed(self.nodes):
            match self.node_types[p]:
                case self.NodeType.DECISION_POINT:
                    V[p] = -inf
                    index = None

                    for i, a in enumerate(self.actions[p]):
                        value = (
                            utility[self.indices[p, a]]
                            + V[self.transitions[p, a]]
                        )

                        if V[p] < value:
                            V[p] = value
                            index = i

                    strategy[p] = np.zeros(len(self.actions[p]))
                    strategy[p][index] = 1
                case self.NodeType.OBSERVATION_POINT:
                    for s in self.signals[p]:
                        V[p] += V[self.transitions[p, s]]

        return strategy, V[self.nodes[0]]

    def sequence_form_best_response(self, utility):
        strategy, value = self.behavioral_best_response(utility)

        return self.behavioral_to_sequence_form(strategy), value

    def behavioral_to_sequence_form(self, behavioral_strategy):
        strategy = np.zeros(len(self.sequences))
        strategy[self.indices[()]] = 1

        for j in self.decision_points:
            p_j = self.parent_sequences[j]

            for i, a in enumerate(self.actions[j]):
                strategy[self.indices[j, a]] = (
                    behavioral_strategy[j][i]
                    * strategy[self.indices[p_j]]
                )

        return strategy

    def counterfactual_utilities(self, behavioral_strategy, utility):
        V = defaultdict(int)

        for p in reversed(self.nodes):
            match self.node_types[p]:
                case self.NodeType.DECISION_POINT:
                    for i, a in enumerate(self.actions[p]):
                        V[p] += (
                            behavioral_strategy[p][i]
                            * (
                                utility[self.indices[p, a]]
                                + V[self.transitions[p, a]]
                            )
                        )
                case self.NodeType.OBSERVATION_POINT:
                    for s in self.signals[p]:
                        V[p] += V[self.transitions[p, s]]

        utilities = {}

        for j in self.decision_points:
            utilities[j] = np.empty(len(self.actions[j]))

            for i, a in enumerate(self.actions[j]):
                utilities[j][i] = (
                    utility[self.indices[j, a]]
                    + V[self.transitions[j, a]]
                )

        return utilities

    def serialize(self):
        raw_data = []

        for parent_edge, p in self.transitions.items():
            raw_data.append(
                {
                    'parent_edge': parent_edge,
                    'node': {'id': p, 'type': self.node_types[p].name},
                },
            )

        return raw_data
