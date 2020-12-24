### Get activations

#### Vanilla
vp_id=3605
va_id=120
vc_id=6849
vs_id=3776

#### Sketch
sp_id=4694
sa_id=1112
sc_id=7715
ss_id=4677


#### Vanilla

echo "Activations: Vanilla Jigsaw"

# Sketch Target: 3811

echo "Sketch"

python get_activations.py --run_id $vs_id --exp_type PACS/vanilla-jigsaw --source art cartoon photo --target sketch --dataset PACS --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vs_id --exp_type PACS/vanilla-jigsaw --source art cartoon photo --target sketch --dataset PACS --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vs_id --exp_type PACS/vanilla-jigsaw --source art cartoon photo --target sketch --dataset PACS --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vs_id --exp_type PACS/vanilla-jigsaw --source art cartoon photo --target sketch --dataset PACS --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Cartoon Target: 6556

echo "Cartoon"

python get_activations.py --run_id $vc_id --exp_type PACS/vanilla-jigsaw --source art sketch photo --target cartoon --dataset PACS --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vc_id --exp_type PACS/vanilla-jigsaw --source art sketch photo --target cartoon --dataset PACS --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vc_id --exp_type PACS/vanilla-jigsaw --source art sketch photo --target cartoon --dataset PACS --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vc_id --exp_type PACS/vanilla-jigsaw --source art sketch photo --target cartoon --dataset PACS --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Art Target: 8913

echo "Art"

python get_activations.py --run_id $va_id --exp_type PACS/vanilla-jigsaw --source sketch cartoon photo --target art --dataset PACS --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $va_id --exp_type PACS/vanilla-jigsaw --source sketch cartoon photo --target art --dataset PACS --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $va_id --exp_type PACS/vanilla-jigsaw --source sketch cartoon photo --target art --dataset PACS --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $va_id --exp_type PACS/vanilla-jigsaw --source sketch cartoon photo --target art --dataset PACS --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Photo Target: 1126

echo "Photo"

python get_activations.py --run_id $vp_id --exp_type PACS/vanilla-jigsaw --source art cartoon sketch --target photo --dataset PACS --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vp_id --exp_type PACS/vanilla-jigsaw --source art cartoon sketch --target photo --dataset PACS --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vp_id --exp_type PACS/vanilla-jigsaw --source art cartoon sketch --target photo --dataset PACS --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vp_id --exp_type PACS/vanilla-jigsaw --source art cartoon sketch --target photo --dataset PACS --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


#### Stylized

echo "Activations: Stylized Jigsaw"

# Sketch Target: 3361

echo "Sketch"

python get_activations.py --run_id $ss_id --stylized --exp_type PACS/stylized-jigsaw --source art cartoon photo --target sketch --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --run_id $ss_id --stylized --exp_type PACS/stylized-jigsaw --source art cartoon photo --target sketch --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --run_id $ss_id --stylized --exp_type PACS/stylized-jigsaw --source art cartoon photo --target sketch --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --run_id $ss_id --stylized --exp_type PACS/stylized-jigsaw --source art cartoon photo --target sketch --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo


# Cartoon Target: 5584

echo "Cartoon"

python get_activations.py --run_id $sc_id --stylized --exp_type PACS/stylized-jigsaw --source art sketch photo --target cartoon --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --run_id $sc_id --stylized --exp_type PACS/stylized-jigsaw --source art sketch photo --target cartoon --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --run_id $sc_id --stylized --exp_type PACS/stylized-jigsaw --source art sketch photo --target cartoon --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --run_id $sc_id --stylized --exp_type PACS/stylized-jigsaw --source art sketch photo --target cartoon --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo


# Art Target: 8057

echo "Art"

python get_activations.py --run_id $sa_id --stylized --exp_type PACS/stylized-jigsaw --source sketch cartoon photo --target art --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --run_id $sa_id --stylized --exp_type PACS/stylized-jigsaw --source sketch cartoon photo --target art --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --run_id $sa_id --stylized --exp_type PACS/stylized-jigsaw --source sketch cartoon photo --target art --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --run_id $sa_id --stylized --exp_type PACS/stylized-jigsaw --source sketch cartoon photo --target art --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo


# Photo Target: 425

echo "Photo"

python get_activations.py --run_id $sp_id --stylized --exp_type PACS/stylized-jigsaw --source art cartoon sketch --target photo --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --run_id $sp_id --stylized --exp_type PACS/stylized-jigsaw --source art cartoon sketch --target photo --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --run_id $sp_id --stylized --exp_type PACS/stylized-jigsaw --source art cartoon sketch --target photo --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --run_id $sp_id --stylized --exp_type PACS/stylized-jigsaw --source art cartoon sketch --target photo --dataset PACS --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo
