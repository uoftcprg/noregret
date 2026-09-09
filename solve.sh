game="turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=7,points_order=descending))"

python solve-noregret.py $game noregret.CUDAKer 1000 data/noregret/gpu.csv
python solve-noregret.py $game noregret.FPKer 1000 data/noregret/cpu.csv
