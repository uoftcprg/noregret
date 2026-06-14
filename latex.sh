python latex.py 'data/gpu/{}.json' 'data/cpp/{}.json' 'data/python/{}.json' 'data/count/{}.json' > tables/speedup.tex
python latex2.py 'data/gpu/{}.json' 'data/cpu/{}.json' 'data/cpp/{}.json' 'data/python/{}.json' > tables/time.tex
python latex3.py 'data/gpu/{}.json' 'data/cpu/{}.json' 'data/cpp/{}.json' 'data/python/{}.json' > tables/space.tex
