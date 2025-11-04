# noregret

No-regret learning dynamics

## Scripts

Replicate the experiment of [Leme, Piliouras, and Schneider (2025)](https://proceedings.neurips.cc/paper_files/paper/2024/file/8b9062635eafbc3677429496a23e424b-Paper-Conference.pdf).

```console
python scripts/leme-piliouras-schneider/main.py
python scripts/leme-piliouras-schneider/main2.py
```

Solve a problem in HW2 of [CMU CSD's 15-888 (2025)](https://www.cs.cmu.edu/~sandholm/cs15-888F25/).

```console
python scripts/15-888/from-15-888.py < path/to/15-888/hw2/cfr_files/rock_paper_superscissors.json > games/extensive-form/15-888/rock-paper-superscissors.json
python scripts/15-888/from-15-888.py < path/to/15-888/hw2/cfr_files/kuhn_poker.json > games/extensive-form/15-888/kuhn-poker.json
python scripts/15-888/from-15-888.py < path/to/15-888/hw2/cfr_files/leduc_poker.json > games/extensive-form/15-888/leduc-poker.json

python scripts/extensive-form-game-statistics.py \
    rock_paper_superscissors games/extensive-form/15-888/rock-paper-superscissors.json \
    kuhn_poker games/extensive-form/15-888/kuhn-poker.json \
    leduc_poker games/extensive-form/15-888/leduc-poker.json \
    > games/extensive-form/15-888/statistics.csv

python scripts/15-888/hw2/main.py
```

Run the BM-CFR experiment.

```console
python scripts/from-open-spiel.py kuhn_poker | python scripts/compress-extensive-form-game.py > games/extensive-form/open-spiel/kuhn-poker.json
python scripts/from-open-spiel.py leduc_poker | python scripts/compress-extensive-form-game.py > games/extensive-form/open-spiel/leduc-poker.json
python scripts/from-open-spiel.py liars_dice | python scripts/compress-extensive-form-game.py > games/extensive-form/open-spiel/liars-dice.json
python scripts/from-open-spiel.py first_sealed_auction | python scripts/compress-extensive-form-game.py > games/extensive-form/open-spiel/first-sealed-auction.json
python scripts/from-open-spiel.py sheriff | python scripts/compress-extensive-form-game.py > games/extensive-form/open-spiel/sheriff.json
python scripts/from-open-spiel.py tiny_bridge_2p | python scripts/compress-extensive-form-game.py > games/extensive-form/open-spiel/2p-tiny-bridge.json
python scripts/from-open-spiel.py tiny_hanabi | python scripts/compress-extensive-form-game.py > games/extensive-form/open-spiel/tiny-hanabi.json

python scripts/extensive-form-game-statistics.py \
    kuhn_poker games/extensive-form/open-spiel/kuhn-poker.json \
    leduc_poker games/extensive-form/open-spiel/leduc-poker.json \
    liars_dice games/extensive-form/open-spiel/liars-dice.json \
    first_sealed_auction games/extensive-form/open-spiel/first-sealed-auction.json \
    sheriff games/extensive-form/open-spiel/sheriff.json \
    tiny_bridge_2p games/extensive-form/open-spiel/2p-tiny-bridge.json \
    tiny_hanabi games/extensive-form/open-spiel/tiny-hanabi.json \
    > games/extensive-form/open-spiel/statistics.csv

python scripts/counterfactual-swap-regret-minimization/main.py > scripts/counterfactual-swap-regret-minimization/average-strategies.jsonl
python scripts/counterfactual-swap-regret-minimization/main2.py < scripts/counterfactual-swap-regret-minimization/average-strategies.jsonl
```

Run the symmetrized game experiment.

```
python scripts/symmetrized-game/main.py
python scripts/symmetrized-game/main2.py
python scripts/symmetrized-game/main3.py
```
