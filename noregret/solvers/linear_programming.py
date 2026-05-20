"""Module for linear programming (LP)."""
from gurobipy import Env, GRB, Model

from noregret.games.extensive_form.games import (
    TwoPlayerZeroSumExtensiveFormGame,
)
from noregret.games.normal_form.games import TwoPlayerZeroSumNormalFormGame


def _lp_2p0s_nfg(game, parameters):
    A = game.payoffs

    with Env(params=parameters) as env:
        m = Model(env=env)
        x = m.addMVar(shape=A.shape[0], name='x')
        u = m.addVar(lb=-GRB.INFINITY, name='u')

        m.setObjective(u, GRB.MAXIMIZE)
        m.addConstr(x.sum() == 1)
        m.addConstr(x @ A >= u)
        m.optimize()

        x = x.X

        m = Model(env=env)
        y = m.addMVar(shape=A.shape[1], name='y')
        v = m.addVar(lb=-GRB.INFINITY, name='v')

        m.setObjective(v, GRB.MINIMIZE)
        m.addConstr(y.sum() == 1)
        m.addConstr(A @ y <= v)
        m.optimize()

        y = y.X

    return x, y


def _lp_2p0s_efg(game, parameters):
    A = game.payoffs
    F = game.row_sequence_form_polytope.constraint_matrix
    f = game.row_sequence_form_polytope.constraint_vector
    G = game.column_sequence_form_polytope.constraint_matrix
    g = game.column_sequence_form_polytope.constraint_vector

    with Env(params=parameters) as env:
        m = Model(env=env)
        x = m.addMVar(shape=A.shape[0], name='x')
        u = m.addMVar(shape=g.shape, lb=-GRB.INFINITY, name='u')

        m.setObjective(g @ u, GRB.MAXIMIZE)
        m.addConstr(F @ x == f)
        m.addConstr(x @ A >= u @ G)
        m.optimize()

        x = x.X

        m = Model(env=env)
        y = m.addMVar(shape=A.shape[1], name='y')
        v = m.addMVar(shape=f.shape, lb=-GRB.INFINITY, name='v')

        m.setObjective(f @ v, GRB.MINIMIZE)
        m.addConstr(G @ y == g)
        m.addConstr(A @ y <= v @ F)
        m.optimize()

        y = y.X

    return x, y


def linear_programming(game, parameters={'OutputFlag': 0}):
    """Solve a game using linear programming.

    Gurobi is used for linear programming.

    :param game: Game.
    :param parameters: Gurobi parameters.
    :return: Nash equilibrium.
    """
    if isinstance(game, TwoPlayerZeroSumNormalFormGame):
        x, y = _lp_2p0s_nfg(game, parameters)
    elif isinstance(game, TwoPlayerZeroSumExtensiveFormGame):
        x, y = _lp_2p0s_efg(game, parameters)
    else:
        raise ValueError('unsupported game type')

    return x, y
