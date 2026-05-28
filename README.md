# GPUGT

GPU parallelizable implementation of counterfactual regret minimization.

## Scripts

Convert games from OpenSpiel.

```console
python scripts/open-spiel-game.py kuhn_poker | python scripts/compress.py > games/kuhn-poker.json
python scripts/open-spiel-game.py leduc_poker | python scripts/compress.py > games/leduc-poker.json
python scripts/open-spiel-game.py liars_dice | python scripts/compress.py > games/liars-dice.json
python scripts/open-spiel-game.py 'battleship(board_height=2,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=3)' | python scripts/compress.py > games/battleship-tiny.json
python scripts/open-spiel-game.py 'battleship(board_height=3,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=3)' | python scripts/compress.py > games/battleship-small.json
python scripts/open-spiel-game.py 'battleship(board_height=4,board_width=4,ship_sizes=[1],ship_values=[1],num_shots=2)' | python scripts/compress.py > games/battleship-medium.json
python scripts/open-spiel-game.py 'battleship(board_height=3,board_width=3,ship_sizes=[1;2],ship_values=[1;1],num_shots=2)' | python scripts/compress.py > games/battleship-large.json
```

Solve games using GPUGT.

```console
python scripts/solve.py \
    games/kuhn-poker.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    1024 \
    -a \
    -e figures/gpugt/exploitabilities/kuhn-poker.pdf \
    -n "Kuhn poker" \
    > data/gpugt/kuhn-poker.json
python scripts/solve.py \
    games/leduc-poker.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    1024 \
    -a \
    -e figures/gpugt/exploitabilities/leduc-poker.pdf \
    -n "Leduc poker" \
    > data/gpugt/leduc-poker.json
python scripts/solve.py \
    games/liars-dice.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    1024 \
    -a \
    -e figures/gpugt/exploitabilities/liars-dice.pdf \
    -n "liar's dice" \
    > data/gpugt/liars-dice.json
python scripts/solve.py \
    games/battleship-tiny.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    8 \
    -a \
    -e figures/gpugt/exploitabilities/battleship-tiny.pdf \
    -n "battleship (tiny)" \
    > data/gpugt/battleship-tiny.json
python scripts/solve.py \
    games/battleship-small.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    8 \
    -a \
    -e figures/gpugt/exploitabilities/battleship-small.pdf \
    -n "battleship (small)" \
    > data/gpugt/battleship-small.json
python scripts/solve.py \
    games/battleship-medium.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    8 \
    -a \
    -e figures/gpugt/exploitabilities/battleship-medium.pdf \
    -n "battleship (medium)" \
    > data/gpugt/battleship-medium.json
python scripts/solve.py \
    games/battleship-large.json \
    gpugt.games.TwoPlayerZeroSumExtensiveFormGame \
    gpugt.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    8 \
    -a \
    -e figures/gpugt/exploitabilities/battleship-large.pdf \
    -n "battleship (large)" \
    > data/gpugt/battleship-large.json
```

Solve games using NoRegret.

```console
python scripts/solve.py \
    games/kuhn-poker.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    1024 \
    -a \
    -e figures/noregret/exploitabilities/kuhn-poker.pdf \
    -n "Kuhn poker" \
    > data/noregret/kuhn-poker.json
python scripts/solve.py \
    games/leduc-poker.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    1024 \
    -a \
    -e figures/noregret/exploitabilities/leduc-poker.pdf \
    -n "Leduc poker" \
    > data/noregret/leduc-poker.json
python scripts/solve.py \
    games/liars-dice.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    1024 \
    -a \
    -e figures/noregret/exploitabilities/liars-dice.pdf \
    -n "liar's dice" \
    > data/noregret/liars-dice.json
python scripts/solve.py \
    games/battleship-tiny.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    8 \
    -a \
    -e figures/noregret/exploitabilities/battleship-tiny.pdf \
    -n "battleship (tiny)" \
    > data/noregret/battleship-tiny.json
python scripts/solve.py \
    games/battleship-small.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    8 \
    -a \
    -e figures/noregret/exploitabilities/battleship-small.pdf \
    -n "battleship (small)" \
    > data/noregret/battleship-small.json
python scripts/solve.py \
    games/battleship-medium.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    8 \
    -a \
    -e figures/noregret/exploitabilities/battleship-medium.pdf \
    -n "battleship (medium)" \
    > data/noregret/battleship-medium.json
python scripts/solve.py \
    games/battleship-large.json \
    noregret.games.TwoPlayerZeroSumExtensiveFormGame \
    noregret.regret_minimizers.CounterfactualRegretMinimization \
    1000 \
    8 \
    -a \
    -e figures/noregret/exploitabilities/battleship-large.pdf \
    -n "battleship (large)" \
    > data/noregret/battleship-large.json
```

Solve games using OpenSpiel (C++).

```console
python scripts/open-spiel-solve.py \
    kuhn_poker \
    pyspiel.CFRSolver \
    1000 \
    1024 \
    -e figures/pyspiel/exploitabilities/kuhn-poker.pdf \
    > data/pyspiel/kuhn-poker.json
python scripts/open-spiel-solve.py \
    leduc_poker \
    pyspiel.CFRSolver \
    1000 \
    1024 \
    -e figures/pyspiel/exploitabilities/leduc-poker.pdf \
    > data/pyspiel/leduc-poker.json
python scripts/open-spiel-solve.py \
    liars_dice \
    pyspiel.CFRSolver \
    1000 \
    1024 \
    -e figures/pyspiel/exploitabilities/liars-dice.pdf \
    > data/pyspiel/liars-dice.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=2,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=3)' \
    pyspiel.CFRSolver \
    1000 \
    8 \
    -e figures/pyspiel/exploitabilities/battleship-tiny.pdf \
    > data/pyspiel/battleship-tiny.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=3,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=3)' \
    pyspiel.CFRSolver \
    1000 \
    8 \
    -e figures/pyspiel/exploitabilities/battleship-small.pdf \
    > data/pyspiel/battleship-small.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=4,board_width=4,ship_sizes=[1],ship_values=[1],num_shots=2)' \
    pyspiel.CFRSolver \
    1000 \
    8 \
    -e figures/pyspiel/exploitabilities/battleship-medium.pdf \
    > data/pyspiel/battleship-medium.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=3,board_width=3,ship_sizes=[1;2],ship_values=[1;1],num_shots=2)' \
    pyspiel.CFRSolver \
    1000 \
    8 \
    -e figures/pyspiel/exploitabilities/battleship-large.pdf \
    > data/pyspiel/battleship-large.json
```

Solve games using OpenSpiel (Python).

```console
python scripts/open-spiel-solve.py \
    kuhn_poker \
    open_spiel.python.algorithms.cfr.CFRSolver \
    1000 \
    8 \
    -e figures/open-spiel/exploitabilities/kuhn-poker.pdf \
    > data/open-spiel/kuhn-poker.json
python scripts/open-spiel-solve.py \
    leduc_poker \
    open_spiel.python.algorithms.cfr.CFRSolver \
    1000 \
    8 \
    -e figures/open-spiel/exploitabilities/leduc-poker.pdf \
    > data/open-spiel/leduc-poker.json
python scripts/open-spiel-solve.py \
    liars_dice \
    open_spiel.python.algorithms.cfr.CFRSolver \
    1000 \
    8 \
    -e figures/open-spiel/exploitabilities/liars-dice.pdf \
    > data/open-spiel/liars-dice.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=2,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=3)' \
    open_spiel.python.algorithms.cfr.CFRSolver \
    1000 \
    8 \
    -e figures/open-spiel/exploitabilities/battleship-tiny.pdf \
    > data/open-spiel/battleship-tiny.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=3,board_width=2,ship_sizes=[2],ship_values=[1],num_shots=3)' \
    open_spiel.python.algorithms.cfr.CFRSolver \
    1000 \
    8 \
    -e figures/open-spiel/exploitabilities/battleship-small.pdf \
    > data/open-spiel/battleship-small.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=4,board_width=4,ship_sizes=[1],ship_values=[1],num_shots=2)' \
    open_spiel.python.algorithms.cfr.CFRSolver \
    1000 \
    8 \
    -e figures/open-spiel/exploitabilities/battleship-medium.pdf \
    > data/open-spiel/battleship-medium.json
python scripts/open-spiel-solve.py \
    'battleship(board_height=3,board_width=3,ship_sizes=[1;2],ship_values=[1;1],num_shots=2)' \
    open_spiel.python.algorithms.cfr.CFRSolver \
    1000 \
    8 \
    -e figures/open-spiel/exploitabilities/battleship-large.pdf \
    > data/open-spiel/battleship-large.json
```

Summarize results.

```console
python scripts/summary.py \
    data/gpugt/kuhn-poker.json \
    data/gpugt/leduc-poker.json \
    data/gpugt/liars-dice.json \
    data/gpugt/battleship-tiny.json \
    data/gpugt/battleship-small.json \
    data/gpugt/battleship-medium.json \
    data/gpugt/battleship-large.json \
    data/noregret/kuhn-poker.json \
    data/noregret/leduc-poker.json \
    data/noregret/liars-dice.json \
    data/noregret/battleship-tiny.json \
    data/noregret/battleship-small.json \
    data/noregret/battleship-medium.json \
    data/noregret/battleship-large.json \
    data/pyspiel/kuhn-poker.json \
    data/pyspiel/leduc-poker.json \
    data/pyspiel/liars-dice.json \
    data/pyspiel/battleship-tiny.json \
    data/pyspiel/battleship-small.json \
    data/pyspiel/battleship-medium.json \
    data/pyspiel/battleship-large.json \
    data/open-spiel/kuhn-poker.json \
    data/open-spiel/leduc-poker.json \
    data/open-spiel/liars-dice.json \
    data/open-spiel/battleship-tiny.json \
    data/open-spiel/battleship-small.json \
    data/open-spiel/battleship-medium.json \
    data/open-spiel/battleship-large.json \
    > data/summary.csv
python scripts/plot.py figures/plot.pdf < data/summary.csv
python scripts/latex.py < data/summary.csv > data/latex.tex

python scripts/summary2.py \
    data/gpugt/kuhn-poker.json \
    data/noregret/kuhn-poker.json \
    data/pyspiel/kuhn-poker.json \
    data/open-spiel/kuhn-poker.json \
    > data/kuhn-poker.csv
python scripts/summary2.py \
    data/gpugt/leduc-poker.json \
    data/noregret/leduc-poker.json \
    data/pyspiel/leduc-poker.json \
    data/open-spiel/leduc-poker.json \
    > data/leduc-poker.csv
python scripts/summary2.py \
    data/gpugt/liars-dice.json \
    data/noregret/liars-dice.json \
    data/pyspiel/liars-dice.json \
    data/open-spiel/liars-dice.json \
    > data/liars-dice.csv
python scripts/summary2.py \
    data/gpugt/battleship-tiny.json \
    data/noregret/battleship-tiny.json \
    data/pyspiel/battleship-tiny.json \
    data/open-spiel/battleship-tiny.json \
    > data/battleship-tiny.csv
python scripts/summary2.py \
    data/gpugt/battleship-small.json \
    data/noregret/battleship-small.json \
    data/pyspiel/battleship-small.json \
    data/open-spiel/battleship-small.json \
    > data/battleship-small.csv
python scripts/summary2.py \
    data/gpugt/battleship-medium.json \
    data/noregret/battleship-medium.json \
    data/pyspiel/battleship-medium.json \
    data/open-spiel/battleship-medium.json \
    > data/battleship-medium.csv
python scripts/summary2.py \
    data/gpugt/battleship-large.json \
    data/noregret/battleship-large.json \
    data/pyspiel/battleship-large.json \
    data/open-spiel/battleship-large.json \
    > data/battleship-large.csv
python scripts/plot2.py \
    figures/plot2.pdf \
    1000 \
    'Kuhn poker' \
    data/kuhn-poker.csv \
    'Leduc poker' \
    data/leduc-poker.csv \
    "Liar's dice" \
    data/liars-dice.csv \
    'Tiny battleship' \
    data/battleship-tiny.csv
python scripts/plot3.py \
    figures/plot3.pdf \
    1000 \
    'Small battleship' \
    data/battleship-small.csv \
    'Medium battleship' \
    data/battleship-medium.csv \
    'Large battleship' \
    data/battleship-large.csv
```

## Citing

If you use GPUGT in your research, please cite our preprint:

```bibtex
@misc{kim2026parallelizingcounterfactualregretminimization,
      title={Parallelizing Counterfactual Regret Minimization}, 
      author={Juho Kim and Tuomas Sandholm},
      year={2026},
      eprint={2605.14277},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.14277}, 
}
```
