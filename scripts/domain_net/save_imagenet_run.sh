
folder_name="DomainNet/vanilla-jigsaw"
num_classes=345
perm=30

# python save_imagenet_model.py --folder_name $folder_name \
# --network resnet18 --n_classes $num_classes --jigsaw_n_classes $perm


#### Vanilla

echo "Activations: Vanilla Jigsaw"

# python get_activations.py --run_id 0 --exp_type $folder_name --source m m m m --target m --dataset DomainNet --generate_for clipart --batch_size 128 --n_classes $num_classes --network resnet18 --jigsaw_n_classes $perm --image_size 222

# python get_activations.py --run_id 0 --exp_type $folder_name --source m m m m --target m --dataset DomainNet --generate_for painting --batch_size 128 --n_classes $num_classes --network resnet18 --jigsaw_n_classes $perm --image_size 222

# python get_activations.py --run_id 0 --exp_type $folder_name --source m m m m --target m --dataset DomainNet --generate_for quickdraw --batch_size 128 --n_classes $num_classes --network resnet18 --jigsaw_n_classes $perm --image_size 222

# python get_activations.py --run_id 0 --exp_type $folder_name --source m m m m --target m --dataset DomainNet --generate_for real --batch_size 128 --n_classes $num_classes --network resnet18 --jigsaw_n_classes $perm --image_size 222

# python get_activations.py --run_id 0 --exp_type $folder_name --source m m m m --target m --dataset DomainNet --generate_for sketch --batch_size 128 --n_classes $num_classes --network resnet18 --jigsaw_n_classes $perm --image_size 222


echo "Linear Evalulation: Vanilla Jigsaw"

# echo "Clipart"

# python LinearEval.py --seed 1 --n_classes $num_classes --run_id 0 --exp_type $folder_name --source m m m m --target m --calc_for clipart

echo "Painting"

python LinearEval.py --seed 1 --n_classes $num_classes --run_id 0 --exp_type $folder_name --source m m m m --target m --calc_for painting

echo "Quick Draw"

python LinearEval.py --seed 1 --n_classes $num_classes --run_id 0 --exp_type $folder_name --source m m m m --target m --calc_for quickdraw

echo "Real"

python LinearEval.py --seed 1 --n_classes $num_classes --run_id 0 --exp_type $folder_name --source m m m m --target m --calc_for real

echo "Sketch"

python LinearEval.py --seed 1 --n_classes $num_classes --run_id 0 --exp_type $folder_name --source m m m m --target m --calc_for sketch
