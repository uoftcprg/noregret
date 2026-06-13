source games.sh

for key in ${!games[@]}; do
	python solve.py ${games[$key]} noregret.CUDAKer noregret.CFR 1000 8 > data/gpu/$key.json
	python solve.py ${games[$key]} noregret.FPKer noregret.CFR 1000 8 > data/cpu/$key.json
	python solve2.py ${games[$key]} pyspiel.CFRSolver 1000 8 > data/cpp/$key.json
	python solve2.py ${games[$key]} open_spiel.python.algorithms.cfr.CFRSolver 1000 8 > data/python/$key.json
done
