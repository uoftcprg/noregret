source games.sh

for key in ${!games[@]}; do
	python solve-noregret.py ${games[$key]} noregret.CUDAKer noregret.CFR 1000 8 data/noregret/cuda/$key.json
	python solve-noregret.py ${games[$key]} noregret.MKLKer noregret.CFR 1000 8 data/noregret/mkl/$key.json
	python solve-noregret.py ${games[$key]} noregret.FPKer noregret.CFR 1000 8 data/noregret/fp/$key.json
	python solve-open-spiel.py ${games[$key]} pyspiel.CFRSolver 1000 8 data/open-spiel/cpp/$key.json
	python solve-open-spiel.py ${games[$key]} open_spiel.python.algorithms.cfr.CFRSolver 1000 8 data/open-spiel/python/$key.json
	python solve-liteefg.py ${games[$key]} utilities 1000 8 data/liteefg/$key.json
done
