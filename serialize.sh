source games.sh

for key in ${!games[@]}; do
	python serialize.py ${games[$key]} data/counts/$key.json data/games/$key.json
done

for key in ${!games2[@]}; do
	python serialize2.py ${games2[$key]} data/counts/$key.json data/games/$key.json
done
