python tabulate-speedups.py 'data/noregret/cuda/{}.json' 'data/open-spiel/cpp/{}.json' 'data/liteefg/{}.json' 'data/count/{}.json' tables/speedups.tex
python tabulate-time.py 'data/noregret/cuda/{}.json' 'data/noregret/mkl/{}.json' 'data/noregret/fp/{}.json' 'data/open-spiel/cpp/{}.json' 'data/open-spiel/python/{}.json' 'data/liteefg/{}.json' tables/time.tex
python tabulate-space.py 'data/noregret/cuda/{}.json' 'data/noregret/mkl/{}.json' 'data/noregret/fp/{}.json' 'data/open-spiel/cpp/{}.json' 'data/open-spiel/python/{}.json' 'data/liteefg/{}.json' tables/space.tex
python tabulate-setup.py 'data/noregret/cuda/{}.json' 'data/liteefg/{}.json' 'data/count/{}.json' tables/setup.tex
