python tabulate-speedups.py 'data/noregret/gpu/{}.json' 'data/open-spiel/cpp/{}.json' 'data/liteefg/{}.json' 'data/count/{}.json' tables/speedups.tex
python tabulate-setup.py 'data/noregret/gpu/{}.json' 'data/liteefg/{}.json' 'data/count/{}.json' tables/setup.tex
python tabulate-time.py 'data/noregret/gpu/{}.json' 'data/noregret/cpu/{}.json' 'data/open-spiel/cpp/{}.json' 'data/open-spiel/python/{}.json' 'data/liteefg/{}.json' tables/time.tex
python tabulate-space.py 'data/noregret/gpu/{}.json' 'data/noregret/cpu/{}.json' 'data/open-spiel/cpp/{}.json' 'data/open-spiel/python/{}.json' 'data/liteefg/{}.json' tables/space.tex
