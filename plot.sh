source games.sh

python plot-times.py $node_count "$branching_factors" 'data/noregret/cuda/{}-{}.json' 'data/noregret/fp/{}-{}.json' figures/times.pdf figures/times.png
python plot-speedups.py $node_count "$branching_factors" 'data/noregret/cuda/{}-{}.json' 'data/noregret/fp/{}-{}.json' figures/speedups.pdf figures/speedups.png
