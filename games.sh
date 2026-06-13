declare -A games=(
	["kuhn-poker"]="kuhn_poker"
	["leduc-poker"]="leduc_poker"
	["liars-dice"]="liars_dice"
	["goofspiel-6"]="turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=6,points_order=descending))"
	["goofspiel-7"]="turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=7,points_order=descending))"
	["battleship-3x2-2-3"]="battleship(board_height=3,board_width=2,ship_sizes=[2],ship_values=[4],num_shots=3)"
	["battleship-3x2-22-3"]="battleship(board_height=3,board_width=2,ship_sizes=[2;2],ship_values=[4;4],num_shots=3)"
)
