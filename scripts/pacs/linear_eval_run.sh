
#### Vanilla
vp_id=8856
va_id=1974
vc_id=5249
vs_id=8565


#### Stylized
sp_id=9476
sa_id=2412
sc_id=5310
ss_id=8620

vanilla_folder_name="PACS/scratch-vanilla-jigsaw"
stylized_folder_name="PACS/scratch-stylized-jigsaw"

echo "Linear Evalulation: Vanilla Jigsaw"

echo "Photo"

python LinearEval.py --seed 1 --n_classes 7 --run_id $vp_id --exp_type $vanilla_folder_name --source art cartoon sketch --target photo --calc_for photo

echo "Art"

python LinearEval.py --seed 1 --n_classes 7 --run_id $va_id --exp_type $vanilla_folder_name --source sketch cartoon photo --target art --calc_for art

echo "Cartoon"

python LinearEval.py --seed 1 --n_classes 7 --run_id $vc_id --exp_type $vanilla_folder_name --source art sketch photo --target cartoon --calc_for cartoon

echo "Sketch"

python LinearEval.py --seed 1 --n_classes 7 --run_id $vs_id --exp_type $vanilla_folder_name --source art cartoon photo --target sketch --calc_for sketch


echo "Linear Evalulation: Stylized Jigsaw"

echo "Photo"

python LinearEval.py --seed 1 --n_classes 7 --run_id $sp_id --exp_type $stylized_folder_name --source art cartoon sketch --target photo --calc_for photo

echo "Art"

python LinearEval.py --seed 1 --n_classes 7 --run_id $sa_id --exp_type $stylized_folder_name --source sketch cartoon photo --target art --calc_for art

echo "Cartoon"

python LinearEval.py --seed 1 --n_classes 7 --run_id $sc_id --exp_type $stylized_folder_name --source art sketch photo --target cartoon --calc_for cartoon

echo "Sketch"

python LinearEval.py --seed 1 --n_classes 7 --run_id $ss_id --exp_type $stylized_folder_name --source art cartoon photo --target sketch --calc_for sketch

