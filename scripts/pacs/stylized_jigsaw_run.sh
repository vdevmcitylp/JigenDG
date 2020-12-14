#!/bin/bash

for lr in 0.1 0.01 0.0001
do

	echo $lr

	python train_jigsaw_evaljig.py --seed 1 --dataset PACS --stylized --jig_only --dataset PACS --folder_name PACS/stylized-jigsaw --source art cartoon photo --target sketch --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
	python train_jigsaw_evaljig.py --seed 1 --dataset PACS --stylized --jig_only --dataset PACS --folder_name PACS/stylized-jigsaw --source art photo sketch --target cartoon --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
	python train_jigsaw_evaljig.py --seed 1 --dataset PACS --stylized --jig_only --dataset PACS --folder_name PACS/stylized-jigsaw --source photo cartoon sketch --target art --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
	python train_jigsaw_evaljig.py --seed 1 --dataset PACS --stylized --jig_only --dataset PACS --folder_name PACS/stylized-jigsaw --source art cartoon sketch --target photo --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9

done