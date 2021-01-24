#!/bin/bash

bias=0.6
perm_set=30
grid=3
num_epochs=150
bs=64
folder_name="OfficeHome/resnet50/scratch-vanilla-jigsaw"


python train_jigsaw_evaljig.py --stylized --jig_only --seed 1 --dataset OfficeHome --folder_name $folder_name --source art clipart realworld --target product --bias_whole_image $bias --epochs $num_epochs --batch_size $bs --n_classes 65 --learning_rate 0.1 --network resnet50 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
python train_jigsaw_evaljig.py --stylized --jig_only --seed 1 --dataset OfficeHome --folder_name $folder_name --source product clipart realworld --target art --bias_whole_image $bias --epochs $num_epochs --batch_size $bs --n_classes 65 --learning_rate 0.1 --network resnet50 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
python train_jigsaw_evaljig.py --stylized --jig_only --seed 1 --dataset OfficeHome --folder_name $folder_name --source product art realworld --target clipart --bias_whole_image $bias --epochs $num_epochs --batch_size $bs --n_classes 65 --learning_rate 0.1 --network resnet50 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9
python train_jigsaw_evaljig.py --stylized --jig_only --seed 1 --dataset OfficeHome --folder_name $folder_name --source product art clipart --target realworld --bias_whole_image $bias --epochs $num_epochs --batch_size $bs --n_classes 65 --learning_rate 0.1 --network resnet50 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw_evaljig.py --jig_only --stylized --seed 1 --dataset OfficeHome --folder_name $folder_name --source art clipart realworld --target product --bias_whole_image $bias --epochs $num_epochs --batch_size $bs --n_classes 65 --learning_rate 0.01 --network resnet50 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9