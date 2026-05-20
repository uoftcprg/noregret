"""Module for kernels."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from noregret.utilities import import_object


@dataclass(repr=False)
class Kernel(ABC):
    """Abstract base class for kernels."""

    def euclidean_projection_on_probability_simplex(self, input_):
        """Euclidean projection of the input on a probability simplex.

        >>> ker = FloatingPointKernel()
        >>> pi = ker.euclidean_projection_on_probability_simplex
        >>> np = ker.numpy
        >>> pi(np.array([0.2, 0.5, 0.3]))
        array([0.2, 0.5, 0.3])
        >>> pi(np.array([0.2, -0.3, 2]))
        array([0., 0., 1.])
        >>> pi(np.array([5, 5, 5, 5]))
        array([0.25, 0.25, 0.25, 0.25])
        >>> pi(np.array([10, 0, 0]))
        array([1., 0., 0.])
        >>> pi(np.array([0.6]))
        array([1.])
        >>> pi(np.array([0, 0, 0, 0, 0]))
        array([0.2, 0.2, 0.2, 0.2, 0.2])

        :param input_: Input to be projected.
        :return: Euclidean projection.
        """
        sorted_input = self.numpy.flip(self.numpy.sort(input_))
        cumsum_sorted_input = sorted_input.cumsum()
        indices = self.numpy.arange(1, input_.size + 1)
        conditions = (sorted_input + (1 - cumsum_sorted_input) / indices) > 0
        rho = self.numpy.where(conditions)[0].max() + 1
        lambda_ = (1 - cumsum_sorted_input[rho - 1]) / rho
        output = (input_ + lambda_).clip(0)

        return output

    def stationary_distribution(self, stochastic_matrix):
        """Calculate a stationary distribution of a right stochastic
        matrix.

        >>> ker = FloatingPointKernel()
        >>> np = ker.numpy
        >>> pi = ker.stationary_distribution
        >>> pi(np.array([[1, 1, 1], [3, 0, 0], [3, 0, 0]]) / 3)
        array([0.6, 0.2, 0.2])

        :param stochastic_matrix: Right stochastic matrix.
        :return: Stationary distribution.
        """
        if not self.numpy.allclose(stochastic_matrix.sum(1), 1):
            raise ValueError('matrix not right stochastic')

        eigenvalues, eigenvectors = self.numpy.linalg.eig(stochastic_matrix.T)
        pi = eigenvectors[:, self.numpy.isclose(eigenvalues, 1)][:, 0]
        pi /= pi.sum()
        pi = pi.real

        assert self.numpy.allclose(pi @ stochastic_matrix, pi)

        return pi

    def standard_basis(self, dimension_count, dimension):
        """Return the standard basis vector for a given dimension in a
        given-dimensional vector space.

        >>> ker = FloatingPointKernel()
        >>> np = ker.numpy
        >>> e = ker.standard_basis
        >>> e(3, 0)
        array([1., 0., 0.])
        >>> e(3, 1)
        array([0., 1., 0.])
        >>> e(3, 2)
        array([0., 0., 1.])

        :param dimension_count: Number of dimensions.
        :param dimension: Dimension.
        :return: Standard basis vector.
        """
        e = self.numpy.zeros(dimension_count)
        e[dimension] = 1

        return e


@dataclass(repr=False)
class ImportedKernel(Kernel, ABC):
    """Abstract base class for imported kernels."""
    numpy_module_path = None
    """NumPy module path."""
    scipy_module_path = None
    """SciPy module path."""
    numpy: Any = field(init=False)
    """NumPy."""
    scipy: Any = field(init=False)
    """SciPy."""

    def __post_init__(self):
        self.numpy = import_object(self.numpy_module_path)
        self.scipy = import_object(self.scipy_module_path)


@dataclass(repr=False)
class FloatingPointKernel(ImportedKernel):
    """Class for floating-point kernels."""
    numpy_module_path = 'numpy'
    scipy_module_path = 'scipy'


@dataclass(repr=False)
class CUDAKernel(ImportedKernel):
    """Class for CUDA kernels."""
    numpy_module_path = 'cupy'
    scipy_module_path = 'cupyx.scipy'

    def __post_init__(self):
        super().__post_init__()

        self.scipy.sparse.csr_array = self.scipy.sparse.csr_matrix


class Serializable(ABC):
    """Abstract base class for serializable objects."""

    @classmethod
    @abstractmethod
    def loads(cls, kernel, raw_data):
        """Load with kernel.

        :param kernel: Kernel.
        :param raw_data: Raw data.
        :return: Loaded data.
        """

    @abstractmethod
    def dumps(self):
        """Dump data.

        :return: Dumped data.
        """
