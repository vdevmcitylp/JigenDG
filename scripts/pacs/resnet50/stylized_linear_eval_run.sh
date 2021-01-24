
#### Stylized
sp_id=6982
sa_id=5892
sc_id=3315
ss_id=3363

folder_name="PACS/scratch-stylized-jigsaw"

echo "Linear Evalulation: Stylized Jigsaw"

echo "Photo"

# python LinearEval.py --seed 1 --n_classes 7 --run_id $sp_id --exp_type $folder_name --source art cartoon sketch --target photo --calc_for photo

echo "Art"

python LinearEval.py --seed 1 --n_classes 7 --run_id $sa_id --exp_type $folder_name --source sketch cartoon photo --target art --calc_for art 

echo "Cartoon"

python LinearEval.py --seed 1 --n_classes 7 --run_id $sc_id --exp_type $folder_name --source art sketch photo --target cartoon --calc_for cartoon

echo "Sketch"

python LinearEval.py --seed 1 --n_classes 7 --run_id $ss_id --exp_type $folder_name --source art cartoon photo --target sketch --calc_for sketch
