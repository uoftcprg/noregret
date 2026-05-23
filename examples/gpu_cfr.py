"""Run GPU-accelerated CFR."""
from sys import stdout

from orjson import dumps, OPT_SERIALIZE_NUMPY
import noregret as nr

CPU_KER = nr.FPKer()
GAME = nr.open_spiel_game(CPU_KER, 'liars_dice')
GPU_KER = nr.CUDAKer()
GAME = nr.to_efg(GPU_KER, GAME)
PARAMETERS = nr.CFR, True, False


def main():
    R_type, alt, pred = PARAMETERS
    R_row = R_type(GPU_KER, GAME.row_sequence_form_polytope)
    R_col = R_type(GPU_KER, GAME.column_sequence_form_polytope)
    x_bar, y_bar = nr.rm(GAME, R_row, R_col, alternation=alt, prediction=pred)
    data = {
        'x_bar': GPU_KER.numpy.asnumpy(x_bar),
        'y_bar': GPU_KER.numpy.asnumpy(y_bar),
        'Exploitability': GAME.exploitability(x_bar, y_bar).item(),
        'Expected utility': GAME.expected_row_utility(x_bar, y_bar).item(),
    }

    stdout.buffer.write(dumps(data, option=OPT_SERIALIZE_NUMPY))


if __name__ == '__main__':
    main()
