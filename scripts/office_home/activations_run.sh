### Get activations

#### Vanilla
echo "Activations: Vanilla Jigsaw"

# Sketch Target: 3811

echo "Sketch"

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 3811 --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 3811 --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 3811 --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 3811 --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Cartoon Target: 6556

echo "Cartoon"

python get_activations.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 6556 --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 6556 --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 6556 --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 6556 --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Art Target: 8913

echo "Art"

python get_activations.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 8913 --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 8913 --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 8913 --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 8913 --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Photo Target: 1126

echo "Photo"

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 1126 --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 1126 --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 1126 --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 1126 --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


#### Stylized

echo "Activations: Stylized Jigsaw"

# Sketch Target: 3361

echo "Sketch"

python get_activations.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 3361 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 3361 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 3361 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 3361 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo


# Cartoon Target: 5584

echo "Cartoon"

python get_activations.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 5584 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 5584 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 5584 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 5584 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo


# Art Target: 8057

echo "Art"

python get_activations.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 8057 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 8057 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 8057 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 8057 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo


# Photo Target: 425

echo "Photo"

python get_activations.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 425 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 425 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 425 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 425 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo
