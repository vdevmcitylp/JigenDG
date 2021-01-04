
lr=0.0001
bias=0.6
perm_set=31

python train_jigsaw_evaljig.py --seed 1 --imagenet --jig_only --dataset PACS --folder_name PACS --source art cartoon photo --target sketch --bias_whole_image $bias --epochs 0 --batch_size 128 --n_classes 7 --learning_rate $lr --network resnet18 --val_size 0.1 --jigsaw_n_classes $perm_set --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9