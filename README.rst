========
NoRegret
========

NoRegret is an open-source software library for no-regret learning dynamics and computational game solving, developed by the Universal, Open, Free, and Transparent Computer Poker Research Group. NoRegret implements an extensive array of regret minimizers and game solvers, and also supports GPU-acceleration. The library can be used in a variety of use cases, from solving games to conducting research in online convex optimization. NoRegret's reliability has been established through extensive doctests and unit tests, achieving 96% code coverage.

Features
--------

* Extensive array of regret minimizers and game solvers.
* High-speed implementations.
* GPU-accleration.

Installation
------------

The NoRegret library requires Python Version 3.12 or above and can be installed using pip:

.. code-block:: bash

   pip install noregret

Usages
------

Example usages of NoRegret is shown below.

Solving Games via Regret minimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code snippet below demonstrates how one can solve games via regret minimization using NoRegret.

.. code-block:: python

   from functools import partial
   from math import inf

   from tqdm import tqdm
   import matplotlib.pyplot as plt
   import noregret as nr
   import pandas as pd
   import seaborn as sns

   KER = nr.FPKer()
   GAMES = {
       'Rock paper superscissors': nr.to_efg(KER, nr.RockPaperSuperscissors(KER)),
       'Kuhn poker': nr.to_efg(KER, nr.open_spiel_game(KER, 'kuhn_poker')),
       'Leduc poker': nr.to_efg(KER, nr.open_spiel_game(KER, 'leduc_poker')),
   }
   PARAMETERS = {
       'CFR': (nr.CFR, False, False),
       'CFR+': (nr.CFR_plus, True, False),
       'DCFR': (nr.DCFR, True, False),
       'PCFR+': (partial(nr.CFR_plus, gamma=2), True, True),
       'PCFR+*': (partial(nr.CFR_plus, gamma=inf), True, True),
   }


   def main():
       for name, game in tqdm(GAMES.items()):
           iterations = []
           exploitabilities = []
           expected_utilities = []
           variants = []

           for variant, (R_type, alt, pred) in tqdm(
                   PARAMETERS.items(),
                   leave=False,
           ):
               R_row = R_type(KER, game.row_sequence_form_polytope)
               R_col = R_type(KER, game.column_sequence_form_polytope)

               def update():
                   t = R_row.iteration_count
                   x_bar = R_row.average_strategy
                   y_bar = R_col.average_strategy
                   epsilon = game.exploitability(x_bar, y_bar)
                   u = game.expected_row_utility(x_bar, y_bar)

                   iterations.append(t)
                   exploitabilities.append(epsilon)
                   expected_utilities.append(u)
                   variants.append(variant)

               nr.rm(
                   game,
                   R_row,
                   R_col,
                   alternation=alt,
                   prediction=pred,
                   update=update,
                   progress_bar={'leave': False},
               )

           data = {
               'Iteration': iterations,
               'Exploitability': exploitabilities,
               'Expected utility': expected_utilities,
               'Variant': variants,
           }
           df = pd.DataFrame(data)

           plt.clf()
           sns.lineplot(df, x='Iteration', y='Exploitability', hue='Variant')
           plt.xscale('log')
           plt.yscale('log')
           plt.title(f'Exploitability in {name}')
           plt.show()

           plt.clf()
           sns.lineplot(df, x='Iteration', y='Expected utility', hue='Variant')
           plt.xscale('log')
           plt.title(f'Expected utility in {name}')
           plt.show()


   if __name__ == '__main__':
       main()

GPU-Accelerated Game Solving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code snippet below demonstrates how one can solve games while leveraging GPU acceleration.

.. code-block:: python

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

Solving Games via Linear Programming
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code snippet below demonstrates how one can solve games via linear programming using NoRegret.

.. code-block:: python

   import noregret as nr

   KER = nr.FPKer()
   GAMES = {
       'Rock paper superscissors': nr.RockPaperSuperscissors(KER),
       'Kuhn poker': nr.to_efg(KER, nr.open_spiel_game(KER, 'kuhn_poker')),
       'Leduc poker': nr.to_efg(KER, nr.open_spiel_game(KER, 'leduc_poker')),
   }


   def main():
       for name, game in GAMES.items():
           x, y = nr.lp(game)
           v = game.expected_row_utility(x, y)

           print(f'{name}:', v)


   if __name__ == '__main__':
       main()

Testing and Validation
----------------------

Run style checks.

.. code-block:: bash

   flake8 examples noregret

Run doctests.

.. code-block:: bash

   shopt -s globstar
   python -m doctest noregret/**/*.py

Run unit tests.

.. code-block:: bash

   python -m unittest

Check coverage.

.. code-block:: bash

   shopt -s globstar
   coverage run -m doctest noregret/**/*.py
   coverage run -a -m unittest
   coverage report -m
   coverage html

Contributing
------------

Contributions are welcome! Please read our Contributing Guide for more information.

License
-------

NoRegret is distributed under the MIT license.

Citing
------

If you use NoRegret in your research, please cite our library:

.. code-block:: bibtex

   @misc{kim2026parallelizingcounterfactualregretminimization,
         title={Parallelizing Counterfactual Regret Minimization},
         author={Juho Kim and Tuomas Sandholm},
         year={2026},
         eprint={2605.14277},
         archivePrefix={arXiv},
         primaryClass={cs.AI},
         url={https://arxiv.org/abs/2605.14277},
   }
