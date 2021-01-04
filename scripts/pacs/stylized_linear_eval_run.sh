
#### Stylized
sp_id=8153
sa_id=4829
sc_id=1554
ss_id=8597


echo "Linear Evalulation: Stylized Jigsaw"

echo "Photo"

python LinearEval.py --seed 1 --n_classes 7 --run_id $sp_id --exp_type PACS/stylized-jigsaw --source art cartoon sketch --target photo 

echo "Art"

python LinearEval.py --seed 1 --n_classes 7 --run_id $sa_id --exp_type PACS/stylized-jigsaw --source sketch cartoon photo --target art 

echo "Cartoon"

python LinearEval.py --seed 1 --n_classes 7 --run_id $sc_id --exp_type PACS/stylized-jigsaw --source art sketch photo --target cartoon 

echo "Sketch"

python LinearEval.py --seed 1 --n_classes 7 --run_id $ss_id --exp_type PACS/stylized-jigsaw --source art cartoon photo --target sketch 
