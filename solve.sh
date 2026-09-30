source games.sh

for branching_factor in $branching_factors; do
	python solve-noregret.py $node_count $branching_factor noregret.CUDAKer float32 noregret.CFR 1000 data/noregret/cuda/$node_count-$branching_factor.json
	python solve-noregret.py $node_count $branching_factor noregret.FPKer float32 noregret.CFR 1000 data/noregret/fp/$node_count-$branching_factor.json
done
