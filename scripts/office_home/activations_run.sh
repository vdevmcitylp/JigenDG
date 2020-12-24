#!/bin/bash

### Get activations

#### Vanilla

vp_id=2016
va_id=319
vc_id=8174
vr_id=5804

#### Stylized

sp_id=6667
sa_id=1385
sc_id=6451
sr_id=1068


#### Vanilla
echo "Activations: Vanilla Jigsaw"

# Product Target: 3811

echo "Product"

python get_activations.py --run_id $vp_id --exp_type OfficeHome/vanilla-jigsaw --source art clipart realworld --target product --dataset OfficeHome --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vp_id --exp_type OfficeHome/vanilla-jigsaw --source art clipart realworld --target product --dataset OfficeHome --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vp_id --exp_type OfficeHome/vanilla-jigsaw --source art clipart realworld --target product --dataset OfficeHome --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vp_id --exp_type OfficeHome/vanilla-jigsaw --source art clipart realworld --target product --dataset OfficeHome --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Art Target: 6556

echo "Art"

python get_activations.py --run_id $va_id --exp_type OfficeHome/vanilla-jigsaw --source product clipart realworld --target art --dataset OfficeHome --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $va_id --exp_type OfficeHome/vanilla-jigsaw --source product clipart realworld --target art --dataset OfficeHome --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $va_id --exp_type OfficeHome/vanilla-jigsaw --source product clipart realworld --target art --dataset OfficeHome --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $va_id --exp_type OfficeHome/vanilla-jigsaw --source product clipart realworld --target art --dataset OfficeHome --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Clipart Target: 8913

echo "Clipart"

python get_activations.py --run_id $vc_id --exp_type OfficeHome/vanilla-jigsaw --source product art realworld --target clipart --dataset OfficeHome --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vc_id --exp_type OfficeHome/vanilla-jigsaw --source product art realworld --target clipart --dataset OfficeHome --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vc_id --exp_type OfficeHome/vanilla-jigsaw --source product art realworld --target clipart --dataset OfficeHome --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vc_id --exp_type OfficeHome/vanilla-jigsaw --source product art realworld --target clipart --dataset OfficeHome --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Realworld Target: 1126

echo "Realworld"

python get_activations.py --run_id $vr_id --exp_type OfficeHome/vanilla-jigsaw --source product art clipart --target realworld --dataset OfficeHome --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vr_id --exp_type OfficeHome/vanilla-jigsaw --source product art clipart --target realworld --dataset OfficeHome --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vr_id --exp_type OfficeHome/vanilla-jigsaw --source product art clipart --target realworld --dataset OfficeHome --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vr_id --exp_type OfficeHome/vanilla-jigsaw --source product art clipart --target realworld --dataset OfficeHome --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


#### Stylized

# echo "Activations: Stylized Jigsaw"

# # Product Target: 3811

# echo "Product"

# python get_activations.py --run_id $sp_id --stylized --exp_type OfficeHome/stylized-jigsaw --source art clipart realworld --target product --dataset OfficeHome --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sp_id --stylized --exp_type OfficeHome/stylized-jigsaw --source art clipart realworld --target product --dataset OfficeHome --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sp_id --stylized --exp_type OfficeHome/stylized-jigsaw --source art clipart realworld --target product --dataset OfficeHome --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sp_id --stylized --exp_type OfficeHome/stylized-jigsaw --source art clipart realworld --target product --dataset OfficeHome --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# # Art Target: 6556

# echo "Art"

# python get_activations.py --run_id $sa_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product clipart realworld --target art --dataset OfficeHome --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sa_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product clipart realworld --target art --dataset OfficeHome --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sa_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product clipart realworld --target art --dataset OfficeHome --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sa_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product clipart realworld --target art --dataset OfficeHome --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# # Clipart Target: 8913

# echo "Clipart"

# python get_activations.py --run_id $sc_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product art realworld --target clipart --dataset OfficeHome --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sc_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product art realworld --target clipart --dataset OfficeHome --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sc_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product art realworld --target clipart --dataset OfficeHome --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sc_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product art realworld --target clipart --dataset OfficeHome --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# # Realworld Target: 1126

# echo "Realworld"

# python get_activations.py --run_id $sr_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product art clipart --target realworld --dataset OfficeHome --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sr_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product art clipart --target realworld --dataset OfficeHome --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sr_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product art clipart --target realworld --dataset OfficeHome --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

# python get_activations.py --run_id $sr_id --stylized --exp_type OfficeHome/stylized-jigsaw --source product art clipart --target realworld --dataset OfficeHome --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222
