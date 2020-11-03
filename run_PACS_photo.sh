# Stylized PACS Domain Generalization Experiments
# Can be run in parallel as separate jobs after running setup_stylized_pacs.sh

# python train_jigsaw.py --stylized --source art cartoon sketch --target photo --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9
# python train_jigsaw.py --stylized --source art cartoon photo --target sketch --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9
# python train_jigsaw.py --stylized --source art sketch photo --target cartoon --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9
# python train_jigsaw.py --stylized --source cartoon sketch photo --target art --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9


# -----------------------------------------------------

# Stylized PACS Jigsaw Solving

# python train_jigsaw_evaljig.py --stylized --source art cartoon sketch --target photo --epochs 100 --jig_only --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1
# python train_jigsaw_evaljig.py --stylized --source photo sketch cartoon --target art --epochs 100 --jig_only --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1
# python train_jigsaw_evaljig.py --stylized --source art sketch photo --target cartoon --epochs 100 --jig_only --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1
# python train_jigsaw_evaljig.py --stylized --source cartoon art photo --target sketch --epochs 100 --jig_only --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1

# -----------------------------------------------------

# Vanilla PACS Jigsaw Solving

# python train_jigsaw_evaljig.py --source art cartoon sketch --target photo --epochs 100 --jig_only --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1
# python train_jigsaw_evaljig.py --source photo sketch cartoon --target art --epochs 100 --jig_only --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1
# python train_jigsaw_evaljig.py --source art sketch photo --target cartoon --epochs 100 --jig_only --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1
# python train_jigsaw_evaljig.py --source cartoon art photo --target sketch --epochs 100 --jig_only --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1

# -----------------------------------------------------

# Stylized Jigsaw Linear Evaluation


# -----------------------------------------------------

# Vanilla PACS Clustering


# -----------------------------------------------------

# Stylized PACS Clustering v1

# -----------------------------------------------------

# Stylized PACS Clustering v2