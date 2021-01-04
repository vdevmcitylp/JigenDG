
folder_name="PACS/vanilla-jigsaw"

python save_imagenet_model.py --folder_name $folder_name \
--network resnet18 --n_classes 7 --jigsaw_n_classes 30 

perm=30

#### Vanilla

# echo "Activations: Vanilla Jigsaw"

python get_activations.py --run_id 0 --exp_type $folder_name --source m m m --target m --dataset PACS --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id 0 --exp_type $folder_name --source m m m --target m --dataset PACS --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id 0 --exp_type $folder_name --source m m m --target m --dataset PACS --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222

python get_activations.py --run_id 0 --exp_type $folder_name --source m m m --target m --dataset PACS --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes $perm --image_size 222



echo "Linear Evalulation: Vanilla Jigsaw"

echo "Photo"

python LinearEval.py --seed 1 --n_classes 7 --run_id 0 --exp_type $folder_name --source m m m --target m --calc_for photo 

echo "Art"

python LinearEval.py --seed 1 --n_classes 7 --run_id 0 --exp_type $folder_name --source m m m --target m --calc_for art

echo "Cartoon"

python LinearEval.py --seed 1 --n_classes 7 --run_id 0 --exp_type $folder_name --source m m m --target m --calc_for cartoon

echo "Sketch"

python LinearEval.py --seed 1 --n_classes 7 --run_id 0 --exp_type $folder_name --source m m m --target m --calc_for sketch
