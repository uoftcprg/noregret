source games.sh

for branching_factor in $branching_factors; do
	python solve-noregret.py $node_count $branching_factor noregret.CUDAKer noregret.CFR 1000 data/noregret/gpu/$node_count-$branching_factor.json
	python solve-noregret.py $node_count $branching_factor noregret.FPKer noregret.CFR 1000 data/noregret/cpu/$node_count-$branching_factor.json
done
