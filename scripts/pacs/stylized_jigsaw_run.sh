#!/bin/bash

bias=0.6
perm_set=30
epochs=200
folder_name="PACS/stylized-jigsaw"
grid=4

for lr in 0.1 0.01 0.001 0.0001
do
	echo $lr

	python train_jigsaw_evaljig.py --seed 1 --stylized --jig_only --grid_size $grid --dataset PACS --folder_name PACS/scratch-stylized-jigsaw --source art cartoon photo --target sketch --bias_whole_image $bias --epochs $epochs --batch_size 128 --n_classes 7 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes $perm_set --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
	python train_jigsaw_evaljig.py --seed 1 --stylized --jig_only --grid_size $grid --dataset PACS --folder_name PACS/scratch-stylized-jigsaw --source art photo sketch --target cartoon --bias_whole_image $bias --epochs $epochs --batch_size 128 --n_classes 7 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes $perm_set --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
	python train_jigsaw_evaljig.py --seed 1 --stylized --jig_only --grid_size $grid --dataset PACS --folder_name PACS/scratch-stylized-jigsaw --source photo cartoon sketch --target art --bias_whole_image $bias --epochs $epochs --batch_size 128 --n_classes 7 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes $perm_set --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
	python train_jigsaw_evaljig.py --seed 1 --stylized --jig_only --grid_size $grid --dataset PACS --folder_name PACS/scratch-stylized-jigsaw --source art cartoon sketch --target photo --bias_whole_image $bias --epochs $epochs --batch_size 128 --n_classes 7 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes $perm_set --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9

done
