
#### Vanilla
vp_id=5663
va_id=8684
vc_id=1918
vs_id=5462


#### Stylized
sp_id=8338
sa_id=1110
sc_id=3935
ss_id=7162


logs_root="/DATA1/vaasudev_narayanan/repositories/JigenDG/logs"

vanilla_folder_name="PACS/scratch-vanilla-jigsaw"
stylized_folder_name="PACS/scratch-stylized-jigsaw"
freeze_layer=3


echo "Linear Evalulation: Vanilla Jigsaw"

echo "Photo"

python LinearEval.py --freeze_layer $freeze_layer --seed 1 --n_classes 7 --run_id $vp_id --exp_type $vanilla_folder_name --source art cartoon sketch --target photo

echo "Art"

python LinearEval.py --freeze_layer $freeze_layer --seed 1 --n_classes 7 --run_id $va_id --exp_type $vanilla_folder_name --source sketch cartoon photo --target art

echo "Cartoon"

python LinearEval.py --freeze_layer $freeze_layer --seed 1 --n_classes 7 --run_id $vc_id --exp_type $vanilla_folder_name --source art sketch photo --target cartoon

echo "Sketch"

python LinearEval.py --freeze_layer $freeze_layer --seed 1 --n_classes 7 --run_id $vs_id --exp_type $vanilla_folder_name --source art cartoon photo --target sketch


echo "Linear Evalulation: Stylized Jigsaw"

echo "Photo"

python LinearEval.py --freeze_layer $freeze_layer --seed 1 --n_classes 7 --run_id $sp_id --exp_type $stylized_folder_name --source art cartoon sketch --target photo

echo "Art"

python LinearEval.py --freeze_layer $freeze_layer --seed 1 --n_classes 7 --run_id $sa_id --exp_type $stylized_folder_name --source sketch cartoon photo --target art

echo "Cartoon"

python LinearEval.py --freeze_layer $freeze_layer --seed 1 --n_classes 7 --run_id $sc_id --exp_type $stylized_folder_name --source art sketch photo --target cartoon

echo "Sketch"

python LinearEval.py --freeze_layer $freeze_layer --seed 1 --n_classes 7 --run_id $ss_id --exp_type $stylized_folder_name --source art cartoon photo --target sketch

