"""Module for sequence-form polytopes."""
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from ordered_set import OrderedSet
from scipy.sparse import lil_array

from noregret.kernels import Kernel


@dataclass
class SequenceFormPolytope:
    """Class for sequence-form polytopes.

    Any vector in behavioral form is of length equal to the number of
    non-empty sequences. Any vector in sequence form is of length equal
    to the number of sequences. By default, values are in sequence form
    unless specified otherwise.
    """
    kernel: Kernel
    """Kernel."""
    actions: dict[str, OrderedSet[str]]
    """Actions."""
    parent_sequences: dict[str, tuple[str, str] | None]
    """Parent sequences for decision points."""
    constraint_matrix: Any = field(init=False)
    """Constraint matrix."""
    constraint_vector: Any = field(init=False)
    """Constraint vector."""
    _A: Any = field(init=False)
    _B: Any = field(init=False)
    _R: Any = field(init=False)
    _C: Any = field(init=False)
    _L_A: list[Any] = field(default_factory=list, init=False)
    _L_B: list[Any] = field(default_factory=list, init=False)
    _L_R: list[Any] = field(default_factory=list, init=False)
    _L_C_A: list[Any] = field(default_factory=list, init=False)
    _L_C_B: list[Any] = field(default_factory=list, init=False)
    _L_C_B2: list[Any] = field(default_factory=list, init=False)

    def __post_init__(self):
        np = self.kernel.numpy
        scipy = self.kernel.scipy
        dtype = self.kernel.data_type
        itype = self.kernel.index_type

        if self.decision_points != self.parent_sequences.keys():
            raise ValueError('inconsistent decision points')

        A = lil_array((self.row_count, self.column_count), dtype=dtype)
        B = lil_array((self.row_count, self.column_count), dtype=dtype)
        A[0, 0] = 1

        for j in self.decision_points:
            p_j = self.parent_sequences[j]
            r = self.row(j)
            c = self.column(p_j)
            B[r, c] = 1

            for a in self.actions[j]:
                c = self.column((j, a))
                A[r, c] = 1

        self._A = scipy.sparse.csr_array(A)
        self._B = scipy.sparse.csr_array(B)
        self.constraint_matrix = self._A - self._B
        self.constraint_vector = self.kernel.standard_basis(self.row_count, 0)
        R = []
        C = []

        for j, a in self.non_empty_sequences:
            R.append(self.row(j))
            C.append(self.column((j, a)))

        self._R = np.array(R, itype)
        self._C = np.array(C, itype)
        children = defaultdict(list)

        for j, p_j in self.parent_sequences.items():
            children[None if p_j is None else p_j[0]].append(j)

        J = children[None]

        while J:
            J_prime = []
            L_R = []
            L_C_A = []
            L_C_B = []
            L_C_B2 = OrderedSet()

            for j in J:
                J_prime.extend(children[j])
                L_R.append(self.row(j))

                p_j = self.parent_sequences[j]

                for a in self.actions[j]:
                    L_C_A.append(self.column((j, a)))
                    L_C_B.append(self.column(p_j))

                L_C_B2.add(self.column(p_j))

            J = J_prime

            self._L_R.append(np.array(L_R, itype))
            self._L_C_A.append(np.array(L_C_A, itype))
            self._L_C_B.append(np.array(L_C_B, itype))
            self._L_C_B2.append(np.array(L_C_B2, itype))

        for L_R, L_C_B in zip(self._L_R, self._L_C_B2):
            self._L_A.append(self._A[L_R])
            self._L_B.append(self._B[L_R][:, L_C_B])

    @cached_property
    def decision_points(self):
        """Return decision points.

        :return: Decision points.
        """
        return OrderedSet(self.actions.keys())

    @cached_property
    def non_empty_sequences(self):
        """Return non-empty sequences.

        :return: Non-empty sequences.
        """
        seqs = OrderedSet()

        for j in self.decision_points:
            for a in self.actions[j]:
                seqs.add((j, a))

        return seqs

    @property
    def row_count(self):
        """Return the number of rows in the constraint matrix.

        :return: Number of rows.
        """
        return len(self.decision_points) + 1

    @property
    def column_count(self):
        """Return the number of columns in the constraint matrix.

        :return: Number of columns.
        """
        return len(self.non_empty_sequences) + 1

    def row(self, decision_point):
        """Return the corresponding row of a given decision point in the
        constraint matrix.

        :param decision_point: Decision point.
        :return: Corresponding row.
        """
        if decision_point is None:
            r = 0
        else:
            r = self.decision_points.index(decision_point) + 1

        return r

    def column(self, sequence):
        """Return the corresponding column of a given sequence in the
        constraint matrix.

        :param sequence: sequence.
        :return: Corresponding column.
        """
        if sequence is None:
            c = 0
        else:
            c = self.non_empty_sequences.index(sequence) + 1

        return c

    @cached_property
    def behavioral_form_uniform_strategy(self):
        """Return the uniform strategy in behavioral form.

        :return: The uniform strategy in behavioral form.
        """
        return ((1 / self._A.sum(1)).ravel() @ self._A)[1:]

    def to_sequence_form(self, behavioral_strategy):
        """Convert a strategy (in behavioral form) to sequence form.

        :param behavioral_strategy: Strategy in behavioral form.
        :return: Strategy in sequence form.
        """
        np = self.kernel.numpy

        if behavioral_strategy.shape != (len(self.non_empty_sequences),):
            raise ValueError('invalid strategy shape')

        strategy = np.r_[1, behavioral_strategy]

        for L_C_A, L_C_B in zip(self._L_C_A, self._L_C_B):
            strategy[L_C_A] *= strategy[L_C_B]

        return strategy

    def _counterfactual_utilities_or_regrets(
            self,
            behavioral_strategy,
            utility,
            normalize,
    ):
        if behavioral_strategy.shape != (len(self.non_empty_sequences),):
            raise ValueError('invalid strategy shape')
        elif utility.shape != (self.column_count,):
            raise ValueError('invalid utility shape')

        A = self._A.copy()
        A[self._R, self._C] = behavioral_strategy
        utility = utility.copy()

        for L_R, L_C_B, L_B in zip(
                self._L_R[::-1],
                self._L_C_B2[::-1],
                self._L_B[::-1],
        ):
            utility[L_C_B] += A[L_R] @ utility @ L_B

        if normalize:
            utility -= A @ utility @ self._A

        return utility[1:]

    def counterfactual_utilities(self, behavioral_strategy, utility):
        """Calculate the counterfactual utilities given a behavioral
        strategy and utility.

        :param behavioral_strategy: Strategy in behavioral form.
        :param utility: Utility.
        :return: Counterfactual utilities.
        """
        return self._counterfactual_utilities_or_regrets(
            behavioral_strategy,
            utility,
            False,
        )

    def counterfactual_regrets(self, behavioral_strategy, utility):
        """Calculate the counterfactual regrets given a behavioral
        strategy and utility.

        :param behavioral_strategy: Strategy in behavioral form.
        :param utility: Utility.
        :return: Counterfactual regrets.
        """
        np = self.kernel.numpy
        dtype = self.kernel.data_type

        if np.isscalar(utility):
            r = np.zeros(len(self.non_empty_sequences), dtype)
        else:
            r = self._counterfactual_utilities_or_regrets(
                behavioral_strategy,
                utility,
                True,
            )

        return r

    def normalize(self, vector):
        """L1-normalize a given vector decision-point-wise.

        :param vector: Vector.
        :return: Normalized vector.
        """
        np = self.kernel.numpy

        if vector.shape != (len(self.non_empty_sequences),):
            raise ValueError('invalid vector shape')

        A = self._A.copy()
        A[self._R, self._C] = vector
        d = A.sum(1).ravel()
        d[np.isclose(d, 0)] = np.nan
        v = ((1 / d) @ A).ravel()[1:]
        m = np.isnan(v)
        v[m] = self.behavioral_form_uniform_strategy[m]

        return v

    def best_response_value(self, utility):
        """Calculate the best response value given a utility.

        The implementation requires the sparse matrix implementation to
        not prune explicit zeros in the Hadamard product.

        :param utility: Utility.
        :return: Best response value.
        """
        if utility.shape != (self.column_count,):
            raise ValueError('invalid shape')

        u = utility.copy()

        for L_A, L_B, L_C_B in zip(
                self._L_A[::-1],
                self._L_B[::-1],
                self._L_C_B2[::-1],
        ):
            v = L_A.multiply(u).max(1, explicit=True).toarray().ravel() @ L_B
            u[L_C_B] += v.ravel()

        return u[0]

    def worst_response_value(self, utility):
        """Calculate the worst response value given a utility.

        The implementation requires the sparse matrix implementation to
        not prune explicit zeros in the Hadamard product.

        :param utility: Utility.
        :return: Worst response value.
        """
        if utility.shape != (self.column_count,):
            raise ValueError('invalid shape')

        u = utility.copy()

        for L_A, L_B, L_C_B in zip(
                self._L_A[::-1],
                self._L_B[::-1],
                self._L_C_B2[::-1],
        ):
            v = L_A.multiply(u).min(1, explicit=True).toarray().ravel() @ L_B
            u[L_C_B] += v.ravel()

        return u[0]
