#!/bin/bash

bias=0.6
perm_set=30
grid=3
num_epochs=150
bs=64
folder_name="Office/resnet50/scratch-vanilla-jigsaw"

python train_jigsaw_evaljig.py --jig_only --seed 1 --dataset Office --folder_name $folder_name --source dslr webcam --target amazon --bias_whole_image $bias --epochs $num_epochs --batch_size $bs --n_classes 31 --learning_rate 0.0001 --network resnet50 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
python train_jigsaw_evaljig.py --jig_only --seed 1 --dataset Office --folder_name $folder_name --source amazon webcam --target dslr --bias_whole_image $bias --epochs $num_epochs --batch_size $bs --n_classes 31 --learning_rate 0.0001 --network resnet50 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
python train_jigsaw_evaljig.py --jig_only --seed 1 --dataset Office --folder_name $folder_name --source amazon dslr --target webcam --bias_whole_image $bias --epochs $num_epochs --batch_size $bs --n_classes 31 --learning_rate 0.0001 --network resnet50 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9