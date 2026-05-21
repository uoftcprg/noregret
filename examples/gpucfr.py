"""Run GPU-accelerated CFR."""
from sys import stdout

from orjson import dumps, OPT_SERIALIZE_NUMPY
import noregret as nr

KERNEL = nr.CUDAKernel()
GAME = nr.to_efg(KERNEL, nr.from_open_spiel('liars_dice'))
PARAMETERS = nr.CFR, True, False


def main():
    R_type, alt, pred = PARAMETERS
    R_row = R_type(KERNEL, GAME.row_sequence_form_polytope)
    R_col = R_type(KERNEL, GAME.column_sequence_form_polytope)
    x_bar, y_bar = nr.regret_minimization(
        GAME,
        R_row,
        R_col,
        alternation=alt,
        prediction=pred,
    )
    data = {
        'x_bar': KERNEL.numpy.asnumpy(x_bar),
        'y_bar': KERNEL.numpy.asnumpy(y_bar),
        'Exploitability': GAME.exploitability(x_bar, y_bar).item(),
        'Expected utility': GAME.expected_row_utility(x_bar, y_bar).item(),
    }

    stdout.buffer.write(dumps(data, option=OPT_SERIALIZE_NUMPY))


if __name__ == '__main__':
    main()
