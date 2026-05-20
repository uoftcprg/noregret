from unittest import main, TestCase

import noregret as nr


class SequenceFormPolytopeTestCase(TestCase):
    KERNELS = (
        nr.FloatingPointKernel(),
        nr.CUDAKernel(),
    )
    ACTIONS = {
        'a': ['b', 'c'],
        'd': ['e'],
        'f': ['g', 'h'],
        'i': ['j', 'k'],
    }
    PARENT_SEQUENCES = {
        'a': None,
        'd': ('a', 'b'),
        'f': ('a', 'c'),
        'i': ('f', 'h'),
    }

    def test_behavioral_form_uniform_strategy(self):
        for kernel in self.KERNELS:
            sfp = nr.SequenceFormPolytope(
                kernel,
                self.ACTIONS,
                self.PARENT_SEQUENCES,
            )
            np = kernel.numpy

            np.testing.assert_array_almost_equal(
                sfp.behavioral_form_uniform_strategy,
                [0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5],
            )

    def test_to_sequence_form(self):
        for kernel in self.KERNELS:
            sfp = nr.SequenceFormPolytope(
                kernel,
                self.ACTIONS,
                self.PARENT_SEQUENCES,
            )
            np = kernel.numpy

            np.testing.assert_array_almost_equal(
                sfp.to_sequence_form(sfp.behavioral_form_uniform_strategy),
                [1, 0.5, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125],
            )
            np.testing.assert_array_almost_equal(
                sfp.to_sequence_form(np.array([0, 1, 1, 0.3, 0.7, 0.5, 0.5])),
                [1, 0, 1, 0, 0.3, 0.7, 0.35, 0.35],
            )

    def test_counterfactual_utilities(self):
        for kernel in self.KERNELS:
            sfp = nr.SequenceFormPolytope(
                kernel,
                self.ACTIONS,
                self.PARENT_SEQUENCES,
            )
            np = kernel.numpy
            b = np.array([0, 1, 1, 0.3, 0.7, 0.5, 0.5])
            u = np.array([0.5, 0, 0, 0, 2, 1, 0, 3])

            np.testing.assert_array_almost_equal(
                sfp.counterfactual_utilities(b, u),
                [0, 2.35, 0, 2, 2.5, 0, 3],
            )

    def test_counterfactual_regrets(self):
        for kernel in self.KERNELS:
            sfp = nr.SequenceFormPolytope(
                kernel,
                self.ACTIONS,
                self.PARENT_SEQUENCES,
            )
            np = kernel.numpy
            b = np.array([0, 1, 1, 0.3, 0.7, 0.5, 0.5])
            u = np.array([0.5, 0, 0, 0, 2, 1, 0, 3])

            np.testing.assert_array_almost_equal(
                sfp.counterfactual_regrets(b, u),
                [-2.35, 0, 0, -0.35, 0.15, -1.5, 1.5],
            )

    def test_normalize(self):
        for kernel in self.KERNELS:
            sfp = nr.SequenceFormPolytope(
                kernel,
                self.ACTIONS,
                self.PARENT_SEQUENCES,
            )
            np = kernel.numpy
            v1 = np.array(np.ones(7))
            v2 = np.array(np.zeros(7))
            b = np.array([0, 1, 1, 0.3, 0.7, 0.5, 0.5])

            np.testing.assert_array_almost_equal(
                sfp.normalize(v1),
                sfp.behavioral_form_uniform_strategy,
            )
            np.testing.assert_array_almost_equal(
                sfp.normalize(v2),
                sfp.behavioral_form_uniform_strategy,
            )
            np.testing.assert_array_almost_equal(sfp.normalize(b), b)

    def test_best_response_value(self):
        for kernel in self.KERNELS:
            sfp = nr.SequenceFormPolytope(
                kernel,
                self.ACTIONS,
                self.PARENT_SEQUENCES,
            )
            np = kernel.numpy
            u = np.array([0.5, 0, 0, 0, 2, 1, 0, 3])

            np.testing.assert_array_almost_equal(
                sfp.best_response_value(u),
                4.5,
            )

            u = np.array([-0.5, -1, -2, 0, 0, 0, 0, 0])

            np.testing.assert_array_almost_equal(
                sfp.best_response_value(u),
                -1.5,
            )

    def test_worst_response_value(self):
        for kernel in self.KERNELS:
            sfp = nr.SequenceFormPolytope(
                kernel,
                self.ACTIONS,
                self.PARENT_SEQUENCES,
            )
            np = kernel.numpy
            u = np.array([0.5, 0, 0, 0, 2, 1, 0, 3])

            np.testing.assert_array_almost_equal(
                sfp.worst_response_value(u),
                0.5,
            )

            u = np.array([0.5, 1, 2, 0, 0, 0, 0, 0])

            np.testing.assert_array_almost_equal(
                sfp.worst_response_value(u),
                1.5,
            )


if __name__ == '__main__':
    main()  # pragma: no cover
