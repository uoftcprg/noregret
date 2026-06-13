source games.sh

for key in ${!games[@]}; do
	python count.py ${games[$key]} > data/count/$key.json
done
