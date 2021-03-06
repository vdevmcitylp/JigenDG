#!/bin/bash

bias=0.6
perm_set=30
folder_name="DomainNet/scratch-vanilla-jigsaw"
grid=3

for lr in 0.1
do
	echo $lr
	echo $grid

	python train_jigsaw_evaljig.py --seed 1 --grid_size $grid --jig_only --dataset DomainNet --folder_name $folder_name --source painting quickdraw real sketch --target clipart --bias_whole_image $bias --epochs 5 --batch_size 128 --n_classes 345 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes $perm_set --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
	python train_jigsaw_evaljig.py --seed 1 --grid_size $grid --jig_only --dataset DomainNet --folder_name $folder_name --source quickdraw real sketch clipart --target painting --bias_whole_image $bias --epochs 200 --batch_size 128 --n_classes 345 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes $perm_set --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
	python train_jigsaw_evaljig.py --seed 1 --grid_size $grid --jig_only --dataset DomainNet --folder_name $folder_name --source real sketch clipart painting --target quickdraw --bias_whole_image $bias --epochs 200 --batch_size 128 --n_classes 345 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes $perm_set --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
	python train_jigsaw_evaljig.py --seed 1 --grid_size $grid --jig_only --dataset DomainNet --folder_name $folder_name --source sketch clipart painting quickdraw --target real --bias_whole_image $bias --epochs 200 --batch_size 128 --n_classes 345 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes $perm_set --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
	python train_jigsaw_evaljig.py --seed 1 --grid_size $grid --jig_only --dataset DomainNet --folder_name $folder_name --source clipart painting quickdraw real --target sketch --bias_whole_image $bias --epochs 200 --batch_size 128 --n_classes 345 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes $perm_set --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9

done

# lr=0.01
# python train_jigsaw_evaljig.py --seed 1 --grid_size 4 --jig_only --dataset PACS --folder_name "PACS/vanilla-jigsaw" --source art cartoon photo --target sketch --bias_whole_image 0.6 --epochs 200 --batch_size 128 --n_classes 7 --learning_rate 0.01 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 224 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9