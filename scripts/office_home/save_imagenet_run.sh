
folder_name="OfficeHome/vanilla-jigsaw"

python save_imagenet_model.py --folder_name $folder_name \
--network resnet18 --n_classes 65 --jigsaw_n_classes 30 

perm=30

#### Vanilla

# echo "Activations: Vanilla Jigsaw"

python get_activations.py --run_id 0 --exp_type $folder_name --source m m m --target m --dataset OfficeHome --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id 0 --exp_type $folder_name --source m m m --target m --dataset OfficeHome --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id 0 --exp_type $folder_name --source m m m --target m --dataset OfficeHome --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id 0 --exp_type $folder_name --source m m m --target m --dataset OfficeHome --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes $perm --image_size 222



echo "Linear Evalulation: Vanilla Jigsaw"

echo "Product"

python LinearEval.py --seed 1 --n_classes 65 --run_id 0 --exp_type $folder_name --source m m m --target m --calc_for product 

echo "Art"

python LinearEval.py --seed 1 --n_classes 65 --run_id 0 --exp_type $folder_name --source m m m --target m --calc_for art

echo "Clipart"

python LinearEval.py --seed 1 --n_classes 65 --run_id 0 --exp_type $folder_name --source m m m --target m --calc_for clipart

echo "Real World"

python LinearEval.py --seed 1 --n_classes 65 --run_id 0 --exp_type $folder_name --source m m m --target m --calc_for realworld
