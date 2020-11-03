# python train_jigsaw.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --source art cartoon photo --target sketch --jig_weight 0.9 --bias_whole_image 0.7



# python train_jigsaw.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --source art cartoon sketch --target photo --jig_weight 0.9 --bias_whole_image 0.6 --epochs 100
# python train_jigsaw.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --source art cartoon photo --target sketch --jig_weight 0.9 --bias_whole_image 0.6 --epochs 100
# python train_jigsaw.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --source art sketch photo --target cartoon --jig_weight 0.9 --bias_whole_image 0.6 --epochs 100
python train_jigsaw.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --source sketch cartoon photo --target art --jig_weight 0.9 --bias_whole_image 0.6 --epochs 100

# python train_jigsaw.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source photo art cartoon --target sketch --jig_weight 0.7 --bias_whole_image 0.9 --image_size 222 --epochs 100
# python train_jigsaw.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source art cartoon sketch --target photo --jig_weight 0.7 --bias_whole_image 0.9 --image_size 222 --epochs 100

# python train_jigsaw_evaljig.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source art cartoon sketch --target photo --jig_weight 0.7 --bias_whole_image 0.9 --image_size 222 --epochs 100 \
# --jig_only
