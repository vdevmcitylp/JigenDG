
echo "Linear Evalulation: Vanilla Jigsaw"

echo "Sketch"

python LinearEval.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 3811

echo "Cartoon"

python LinearEval.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 6556

echo "Art"

python LinearEval.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 8913

echo "Photo"

python LinearEval.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 1126

echo "Linear Evalulation: Stylized Jigsaw"

echo "Sketch"

python LinearEval.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 3361

echo "Cartoon"

python LinearEval.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 5584

echo "Art"

python LinearEval.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 8057

echo "Photo"

python LinearEval.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 425
