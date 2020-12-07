
### Stylized Office Home Experiments

<hr>

### Jigsaw Solving

#### Vanilla 

Realworld Target

python train_jigsaw_evaljig.py --jig_only --folder_name vanilla-jigsaw --source art product clipart  --target realworld  --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9


Art Target

python train_jigsaw_evaljig.py --jig_only --folder_name vanilla-jigsaw --source product realworld clipart --target art  --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9

Clipart Target

python train_jigsaw_evaljig.py --jig_only --folder_name vanilla-jigsaw --source art product realworld --target clipart --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9

Product Target

python train_jigsaw_evaljig.py --jig_only --folder_name vanilla-jigsaw --source art  realworld clipart --target product --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9

<hr>

#### Stylized 

Realworld Target
 
python train_jigsaw_evaljig.py --jig_only --folder_name stylized-jigsaw --source art product clipart  --target realworld  --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9 --stylized


Art Target

python train_jigsaw_evaljig.py --jig_only --folder_name stylized-jigsaw --source product realworld clipart --target art  --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9 --stylized

Clipart Target

python train_jigsaw_evaljig.py --jig_only --folder_name stylized-jigsaw --source art product realworld --target clipart --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9 --stylized

Product Target

python train_jigsaw_evaljig.py --jig_only --folder_name stylized-jigsaw --source art  realworld clipart --target product --bias_whole_image 0.6 --epochs 100 --batch_size 128 --n_classes 65 --learning_rate 0.001 --network resnet18 --val_size 0.1 --jigsaw_n_classes 30 --train_all --image_size 222 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --jig_weight 0.9 --stylized

<hr>


### Get Activations

#### Vanilla

Realworld Target

python get_activations.py --exp_type vanilla-jigsaw --source  clipart art product --target realworld --run_id 3105 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for realworld

python get_activations.py --exp_type vanilla-jigsaw --source  clipart art product --target realworld --run_id 3105 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for clipart

python get_activations.py --exp_type vanilla-jigsaw --source  clipart art product --target realworld --run_id 3105 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for art

python get_activations.py --exp_type vanilla-jigsaw --source  clipart art product --target realworld --run_id 3105 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for product

<hr>


Art Target

python get_activations.py --exp_type vanilla-jigsaw --source  realworld product clipart --target art  --run_id 3176 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for realworld

python get_activations.py --exp_type vanilla-jigsaw --source  realworld product clipart --target art --run_id 3176 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for clipart

python get_activations.py --exp_type vanilla-jigsaw --source  realworld product clipart --target art --run_id 3176 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for art

python get_activations.py --exp_type vanilla-jigsaw --source  realworld product clipart --target art --run_id 3176 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for product

<hr>

Clipart Target

python get_activations.py --exp_type vanilla-jigsaw --source  realworld art product --target clipart  --run_id 3067 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for realworld

python get_activations.py --exp_type vanilla-jigsaw --source  realworld art product --target clipart --run_id 3067 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for clipart

python get_activations.py --exp_type vanilla-jigsaw --source  realworld art product --target clipart --run_id 3067 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for art

python get_activations.py --exp_type vanilla-jigsaw --source  realworld art product --target clipart --run_id 3067 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for product

<hr>

Product Target

python get_activations.py --exp_type vanilla-jigsaw --source  realworld art clipart --target product  --run_id 3152 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for realworld

python get_activations.py --exp_type vanilla-jigsaw --source  realworld art clipart --target product --run_id 3152 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for clipart

python get_activations.py --exp_type vanilla-jigsaw --source  realworld art clipart --target product --run_id 3152 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for art

python get_activations.py --exp_type vanilla-jigsaw --source  realworld art clipart --target product --run_id 3152 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for product

<hr>

#### Stylized 

Realworld Target

python get_activations.py --exp_type stylized-jigsaw --source  clipart art product --target realworld --run_id 6229 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for realworld --stylized

python get_activations.py --exp_type stylized-jigsaw --source  clipart art product --target realworld --run_id 6229 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for clipart --stylized

python get_activations.py --exp_type stylized-jigsaw --source  clipart art product --target realworld --run_id 6229 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for art --stylized

python get_activations.py --exp_type stylized-jigsaw --source  clipart art product --target realworld --run_id 6229 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for product --stylized

<hr>

Art Target

python get_activations.py --exp_type stylized-jigsaw --source  realworld product clipart --target art  --run_id 6125 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for realworld --stylized

python get_activations.py --exp_type stylized-jigsaw --source  realworld product clipart --target art --run_id 6125 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for clipart --stylized

python get_activations.py --exp_type stylized-jigsaw --source  realworld product clipart --target art --run_id 6125 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for art --stylized

python get_activations.py --exp_type stylized-jigsaw --source  realworld product clipart --target art --run_id 6125 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for product --stylized

<hr>

Clipart Target

python get_activations.py --exp_type stylized-jigsaw --source  realworld art product --target clipart  --run_id 6254 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for realworld --stylized

python get_activations.py --exp_type stylized-jigsaw --source  realworld art product --target clipart --run_id 6254 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for clipart --stylized

python get_activations.py --exp_type stylized-jigsaw --source  realworld art product --target clipart --run_id 6254 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for art --stylized

python get_activations.py --exp_type stylized-jigsaw --source  realworld art product --target clipart --run_id 6254 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for product --stylized

<hr>

Product Target

python get_activations.py --exp_type stylized-jigsaw --source  realworld art clipart --target product  --run_id 6477 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for realworld --stylized

python get_activations.py --exp_type stylized-jigsaw --source  realworld art clipart --target product --run_id 6477 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for clipart --stylized

python get_activations.py --exp_type stylized-jigsaw --source  realworld art clipart --target product --run_id 6477 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for art --stylized

python get_activations.py --exp_type stylized-jigsaw --source  realworld art clipart --target product --run_id 6477 --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222  --generate_for product --stylized

<hr>


### Linear Evaluation

#### Vanilla 

python LinearEval.py --exp_type vanilla-jigsaw --source art clipart product --target realworld --run_id 3105


python LinearEval.py --exp_type vanilla-jigsaw --source product clipart realworld --target art  --run_id 3176


python LinearEval.py --exp_type vanilla-jigsaw --source art product realworld --target  clipart --run_id 3067


python LinearEval.py --exp_type vanilla-jigsaw --source art clipart realworld --target product --run_id 3152

<hr>

#### Stylized 

python LinearEval.py --exp_type stylized-jigsaw --source art clipart product --target realworld --run_id 6229 

python LinearEval.py --exp_type stylized-jigsaw --source product clipart realworld --target art  --run_id 6125 

python LinearEval.py --exp_type stylized-jigsaw --source art product realworld --target  clipart --run_id 6254 

python LinearEval.py --exp_type stylized-jigsaw --source art clipart realworld --target product --run_id 6477 

<hr>

### Clustering

#### Vanilla Source Clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art clipart product --target realworld --run_id 3105 --source_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art realworld product --target clipart --run_id 3067 --source_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source realworld clipart product --target art --run_id 3176 --source_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art clipart realworld --target product --run_id 3152 --source_clustering

<hr>

#### Vanilla Target Clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art clipart product --target realworld --run_id 3105 --target_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art realworld product --target clipart --run_id 3067 --target_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source realworld clipart product --target art --run_id 3176 --target_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art clipart realworld --target product --run_id 3152 --target_clustering

<hr>

#### Stylized Source Clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art clipart product --target realworld --run_id 6477 --source_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art realworld product --target clipart --run_id 6254 --source_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source realworld clipart product --target art --run_id 6125 --source_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art clipart realworld --target product --run_id 6229 --source_clustering

<hr>

#### Stylized Target Clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art clipart product --target realworld --run_id 6477 --target_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art realworld product --target clipart --run_id 6254 --target_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source realworld clipart product --target art --run_id 6125 --target_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art clipart realworld --target product --run_id 6229 --target_clustering