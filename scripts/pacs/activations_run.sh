### Get activations

#### Vanilla
vp_id=4361
va_id=1079
vc_id=7830
vs_id=4846


#### Stylized
sp_id=7853
sa_id=4284
sc_id=960
ss_id=7915


vanilla_folder_name="PACS/vanilla-jigsaw"
stylized_folder_name="PACS/stylized-jigsaw"
perm=1000

#### Vanilla

echo "Activations: Vanilla Jigsaw"

# Sketch Target: 3811

echo "Sketch"

python get_activations.py --run_id $vs_id --exp_type $vanilla_folder_name --source art cartoon photo --target sketch --dataset PACS --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $vs_id --exp_type $vanilla_folder_name --source art cartoon photo --target sketch --dataset PACS --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $vs_id --exp_type $vanilla_folder_name --source art cartoon photo --target sketch --dataset PACS --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $vs_id --exp_type $vanilla_folder_name --source art cartoon photo --target sketch --dataset PACS --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222


# Cartoon Target: 6556

echo "Cartoon"

python get_activations.py --run_id $vc_id --exp_type $vanilla_folder_name --source art sketch photo --target cartoon --dataset PACS --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $vc_id --exp_type $vanilla_folder_name --source art sketch photo --target cartoon --dataset PACS --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $vc_id --exp_type $vanilla_folder_name --source art sketch photo --target cartoon --dataset PACS --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $vc_id --exp_type $vanilla_folder_name --source art sketch photo --target cartoon --dataset PACS --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222


# Art Target: 8913

echo "Art"

python get_activations.py --run_id $va_id --exp_type $vanilla_folder_name --source sketch cartoon photo --target art --dataset PACS --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $va_id --exp_type $vanilla_folder_name --source sketch cartoon photo --target art --dataset PACS --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $va_id --exp_type $vanilla_folder_name --source sketch cartoon photo --target art --dataset PACS --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $va_id --exp_type $vanilla_folder_name --source sketch cartoon photo --target art --dataset PACS --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222


# Photo Target: 1126

echo "Photo"

python get_activations.py --run_id $vp_id --exp_type $vanilla_folder_name --source art cartoon sketch --target photo --dataset PACS --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $vp_id --exp_type $vanilla_folder_name --source art cartoon sketch --target photo --dataset PACS --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $vp_id --exp_type $vanilla_folder_name --source art cartoon sketch --target photo --dataset PACS --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id $vp_id --exp_type $vanilla_folder_name --source art cartoon sketch --target photo --dataset PACS --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222


### Stylized

echo "Activations: Stylized Jigsaw"

# Sketch Target: 3361

echo "Sketch"

python get_activations.py --run_id $ss_id --stylized --exp_type $stylized_folder_name --source art cartoon photo --target sketch --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for sketch

python get_activations.py --run_id $ss_id --stylized --exp_type $stylized_folder_name --source art cartoon photo --target sketch --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for cartoon

python get_activations.py --run_id $ss_id --stylized --exp_type $stylized_folder_name --source art cartoon photo --target sketch --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for art

python get_activations.py --run_id $ss_id --stylized --exp_type $stylized_folder_name --source art cartoon photo --target sketch --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for photo


# Cartoon Target: 5584

echo "Cartoon"

python get_activations.py --run_id $sc_id --stylized --exp_type $stylized_folder_name --source art sketch photo --target cartoon --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for sketch

python get_activations.py --run_id $sc_id --stylized --exp_type $stylized_folder_name --source art sketch photo --target cartoon --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for cartoon

python get_activations.py --run_id $sc_id --stylized --exp_type $stylized_folder_name --source art sketch photo --target cartoon --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for art

python get_activations.py --run_id $sc_id --stylized --exp_type $stylized_folder_name --source art sketch photo --target cartoon --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for photo


# Art Target: 8057

echo "Art"

python get_activations.py --run_id $sa_id --stylized --exp_type $stylized_folder_name --source sketch cartoon photo --target art --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for sketch

python get_activations.py --run_id $sa_id --stylized --exp_type $stylized_folder_name --source sketch cartoon photo --target art --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for cartoon

python get_activations.py --run_id $sa_id --stylized --exp_type $stylized_folder_name --source sketch cartoon photo --target art --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for art

python get_activations.py --run_id $sa_id --stylized --exp_type $stylized_folder_name --source sketch cartoon photo --target art --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for photo


# Photo Target: 425

echo "Photo"

python get_activations.py --run_id $sp_id --stylized --exp_type $stylized_folder_name --source art cartoon sketch --target photo --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for sketch

python get_activations.py --run_id $sp_id --stylized --exp_type $stylized_folder_name --source art cartoon sketch --target photo --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for cartoon

python get_activations.py --run_id $sp_id --stylized --exp_type $stylized_folder_name --source art cartoon sketch --target photo --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for art

python get_activations.py --run_id $sp_id --stylized --exp_type $stylized_folder_name --source art cartoon sketch --target photo --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222 --stylized --generate_for photo
