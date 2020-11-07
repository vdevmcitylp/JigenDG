### Stylized PACS Domain Generalization Experiments

Can be run in parallel as separate jobs after running setup_stylized_pacs.sh

python train_jigsaw.py --stylized --source art cartoon photo --target sketch --bias_whole_image 0.6 --folder_name StylizedDG --epochs 100  --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw.py --stylized --source art photo sketch --target cartoon --bias_whole_image 0.6 --folder_name StylizedDG --epochs 100  --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw.py --stylized --source photo cartoon sketch --target art --bias_whole_image 0.6 --folder_name StylizedDG --epochs 100  --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw.py --stylized --source art cartoon sketch --target photo --bias_whole_image 0.6 --folder_name StylizedDG --epochs 100  --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9

<hr>

### Jigsaw Solving

#### Vanilla

python train_jigsaw_evaljig.py --jig_only --folder_name vanilla-jigsaw --source art cartoon photo --target sketch --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw_evaljig.py --jig_only --folder_name vanilla-jigsaw --source art photo sketch --target cartoon --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw_evaljig.py --jig_only --folder_name vanilla-jigsaw --source photo cartoon sketch --target art --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw_evaljig.py --jig_only --folder_name vanilla-jigsaw --source art cartoon sketch --target photo --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9


#### Stylized

python train_jigsaw_evaljig.py --stylized --jig_only --folder_name stylized-jigsaw --source art cartoon photo --target sketch --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw_evaljig.py --stylized --jig_only --folder_name stylized-jigsaw --source art photo sketch --target cartoon --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw_evaljig.py --stylized --jig_only --folder_name stylized-jigsaw --source photo cartoon sketch --target art --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9

python train_jigsaw_evaljig.py --stylized --jig_only --folder_name stylized-jigsaw --source art cartoon sketch --target photo --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --jig_weight 0.9

<hr>

### Get activations

#### Vanilla

Sketch Target: 9095

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 9095 --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 9095 --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 9095 --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 9095 --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


<hr>

Cartoon Target: 9132

python get_activations.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 9132 --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 9132 --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 9132 --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 9132 --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


<hr>

Art Target: 2126

python get_activations.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 2126 --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 2126 --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 2126 --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 2126 --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


<hr>

Photo Target: 2138

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 2138 --generate_for sketch --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 2138 --generate_for cartoon --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 2138 --generate_for art --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 2138 --generate_for photo --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222


<hr>

#### Stylized

Sketch Target: 1530

python get_activations.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 1530 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 1530 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 1530 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 1530 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo

<hr>

Cartoon Target

python get_activations.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 1587 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 1587 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 1587 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 1587 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo

<hr>

Art Target

python get_activations.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 6093 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 6093 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 6093 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 6093 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo

<hr>

Photo Target

python get_activations.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 6055 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for sketch

python get_activations.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 6055 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for cartoon

python get_activations.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 6055 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for art

python get_activations.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 6055 --batch_size 128 --n_classes 7 --network resnet18 --jigsaw_n_classes 30 --image_size 222 --stylized --generate_for photo

<hr>

### Linear Evaluation

#### Vanilla

python LinearEval.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 8561

python LinearEval.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 8542

python LinearEval.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 5628

python LinearEval.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 5513

#### Stylized

python LinearEval.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 9401

python LinearEval.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 9428

python LinearEval.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 2714

python LinearEval.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 2697

<hr>

### Clustering Evaluation

#### Vanilla Source Clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 8561 --source_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 8542 --source_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 5628 --source_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 5513 --source_clustering


#### Vanilla Target Clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 8561 --target_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 8542 --target_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 5628 --target_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 5513 --target_clustering


#### Stylized Source Clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 9401 --source_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 9428 --source_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 2714 --source_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 2697 --source_clustering


#### Stylized Target Clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 9401 --target_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 9428 --target_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 2714 --target_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 2697 --target_clustering

