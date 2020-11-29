
# python train_jigsaw.py --deep_all --stylized --source art cartoon photo --target sketch --bias_whole_image 0.6 --folder_name StylizedDG --epochs 100  --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

# python train_jigsaw.py --deep_all --stylized --source art photo sketch --target cartoon --bias_whole_image 0.6 --folder_name StylizedDG --epochs 100  --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

# python train_jigsaw.py --deep_all --stylized --source photo cartoon sketch --target art --bias_whole_image 0.6 --folder_name StylizedDG --epochs 100  --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

# python train_jigsaw.py --deep_all --stylized --source art cartoon sketch --target photo --bias_whole_image 0.6 --folder_name StylizedDG --epochs 100  --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

# DeepAll

# Seed 1

python train_jigsaw.py --seed 1 --deep_all --source product realworld clipart --target art --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

python train_jigsaw.py --seed 1 --deep_all --source product realworld art --target clipart --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

python train_jigsaw.py --seed 1 --deep_all --source realworld art clipart --target product --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

python train_jigsaw.py --seed 1 --deep_all --source art product clipart --target realworld --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

# Seed 2

python train_jigsaw.py --seed 2 --deep_all --source product realworld clipart --target art --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

python train_jigsaw.py --seed 2 --deep_all --source product realworld art --target clipart --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

python train_jigsaw.py --seed 2 --deep_all --source realworld art clipart --target product --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

python train_jigsaw.py --seed 2 --deep_all --source art product clipart --target realworld --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

# Seed 3

python train_jigsaw.py --seed 3 --deep_all --source product realworld clipart --target art --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

python train_jigsaw.py --seed 3 --deep_all --source product realworld art --target clipart --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

python train_jigsaw.py --seed 3 --deep_all --source realworld art clipart --target product --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1

python train_jigsaw.py --seed 3 --deep_all --source art product clipart --target realworld --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --dataset OfficeHome --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1