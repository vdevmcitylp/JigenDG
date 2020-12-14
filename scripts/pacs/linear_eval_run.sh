
echo "Linear Evalulation: Vanilla Jigsaw"

echo "Sketch"

python LinearEval.py --seed 1 --run_id 3811 --exp_type PACS/vanilla-jigsaw --source art cartoon photo --target sketch

echo "Cartoon"

python LinearEval.py --seed 1 --run_id 6556 --exp_type PACS/vanilla-jigsaw --source art sketch photo --target cartoon

echo "Art"

python LinearEval.py --seed 1 --run_id 8913 --exp_type PACS/vanilla-jigsaw --source sketch cartoon photo --target art

echo "Photo"

python LinearEval.py --seed 1 --run_id 1126 --exp_type PACS/vanilla-jigsaw --source art cartoon sketch --target photo


echo "Linear Evalulation: Stylized Jigsaw"

echo "Sketch"

python LinearEval.py --seed 1 --run_id 3361 --exp_type PACS/stylized-jigsaw --source art cartoon photo --target sketch 

echo "Cartoon"

python LinearEval.py --seed 1 --run_id 5584 --exp_type PACS/stylized-jigsaw --source art sketch photo --target cartoon 

echo "Art"

python LinearEval.py --seed 1 --run_id 8057 --exp_type PACS/stylized-jigsaw --source sketch cartoon photo --target art 

echo "Photo"

python LinearEval.py --seed 1 --run_id 425 --exp_type PACS/stylized-jigsaw --source art cartoon sketch --target photo 
