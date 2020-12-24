
echo "Linear Evalulation: Vanilla Jigsaw"


echo "Photo"

python LinearEval.py --seed 1 --run_id 3605 --exp_type PACS/vanilla-jigsaw --source art cartoon sketch --target photo

echo "Art"

python LinearEval.py --seed 1 --run_id 120 --exp_type PACS/vanilla-jigsaw --source sketch cartoon photo --target art

echo "Cartoon"

python LinearEval.py --seed 1 --run_id 6849 --exp_type PACS/vanilla-jigsaw --source art sketch photo --target cartoon

echo "Sketch"

python LinearEval.py --seed 1 --run_id 3776 --exp_type PACS/vanilla-jigsaw --source art cartoon photo --target sketch


echo "Linear Evalulation: Stylized Jigsaw"

echo "Photo"

python LinearEval.py --seed 1 --run_id 4694 --exp_type PACS/stylized-jigsaw --source art cartoon sketch --target photo 

echo "Art"

python LinearEval.py --seed 1 --run_id 1112 --exp_type PACS/stylized-jigsaw --source sketch cartoon photo --target art 

echo "Cartoon"

python LinearEval.py --seed 1 --run_id 7715 --exp_type PACS/stylized-jigsaw --source art sketch photo --target cartoon 

echo "Sketch"

python LinearEval.py --seed 1 --run_id 4677 --exp_type PACS/stylized-jigsaw --source art cartoon photo --target sketch 
