source games.sh

python plot-times.py $node_count "$branching_factors" 'data/noregret/gpu/{}-{}.json' 'data/noregret/cpu/{}-{}.json' figures/times.pdf figures/times.png
python plot-speedups.py $node_count "$branching_factors" 'data/noregret/gpu/{}-{}.json' 'data/noregret/cpu/{}-{}.json' figures/speedups.pdf figures/speedups.png
