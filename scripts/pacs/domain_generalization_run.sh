
python train_jigsaw.py --stylized --source art product clipart --target realworld --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw.py --stylized --source product realworld clipart --target art --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw.py --stylized --source product realworld art --target clipart --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw.py --stylized --source realworld art clipart --target product --bias_whole_image 0.6 --folder_name OfficeHome/DeepAll --epochs 100  --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9


