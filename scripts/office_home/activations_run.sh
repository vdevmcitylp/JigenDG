#!/bin/bash

### Get activations

#### Vanilla

vp_id=3811
va_id=6556
vc_id=8913
vr_id=1126

#### Stylized

sp_id=3811
sa_id=6556
sc_id=8913
sr_id=1126


#### Vanilla
echo "Activations: Vanilla Jigsaw"

# Product Target: 3811

echo "Product"

python get_activations.py --run_id $vp_id --exp_type OfficeHome/vanilla-jigsaw --source art clipart realworld --target product --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vp_id --exp_type OfficeHome/vanilla-jigsaw --source art clipart realworld --target product --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vp_id --exp_type OfficeHome/vanilla-jigsaw --source art clipart realworld --target product --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vp_id --exp_type OfficeHome/vanilla-jigsaw --source art clipart realworld --target product --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Art Target: 6556

echo "Art"

python get_activations.py --run_id $va_id --exp_type OfficeHome/vanilla-jigsaw --source product clipart realworld --target art --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $va_id --exp_type OfficeHome/vanilla-jigsaw --source product clipart realworld --target art --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $va_id --exp_type OfficeHome/vanilla-jigsaw --source product clipart realworld --target art --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $va_id --exp_type OfficeHome/vanilla-jigsaw --source product clipart realworld --target art --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Clipart Target: 8913

echo "Clipart"

python get_activations.py --run_id $vc_id --exp_type OfficeHome/vanilla-jigsaw --source product art realworld --target clipart --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vc_id --exp_type OfficeHome/vanilla-jigsaw --source product art realworld --target clipart --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vc_id --exp_type OfficeHome/vanilla-jigsaw --source product art realworld --target clipart --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vc_id --exp_type OfficeHome/vanilla-jigsaw --source product art realworld --target clipart --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Realworld Target: 1126

echo "Realworld"

python get_activations.py --run_id $vr_id --exp_type OfficeHome/vanilla-jigsaw --source product art clipart --target realworld --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vr_id --exp_type OfficeHome/vanilla-jigsaw --source product art clipart --target realworld --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vr_id --exp_type OfficeHome/vanilla-jigsaw --source product art clipart --target realworld --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $vr_id --exp_type OfficeHome/vanilla-jigsaw --source product art clipart --target realworld --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


#### Stylized

echo "Activations: Stylized Jigsaw"

# Product Target: 3811

echo "Product"

python get_activations.py --run_id $sp_id --exp_type OfficeHome/stylized-jigsaw --source art clipart realworld --target product --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sp_id --exp_type OfficeHome/stylized-jigsaw --source art clipart realworld --target product --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sp_id --exp_type OfficeHome/stylized-jigsaw --source art clipart realworld --target product --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sp_id --exp_type OfficeHome/stylized-jigsaw --source art clipart realworld --target product --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Art Target: 6556

echo "Art"

python get_activations.py --run_id $sa_id --exp_type OfficeHome/stylized-jigsaw --source product clipart realworld --target art --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sa_id --exp_type OfficeHome/stylized-jigsaw --source product clipart realworld --target art --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sa_id --exp_type OfficeHome/stylized-jigsaw --source product clipart realworld --target art --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sa_id --exp_type OfficeHome/stylized-jigsaw --source product clipart realworld --target art --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Clipart Target: 8913

echo "Clipart"

python get_activations.py --run_id $sc_id --exp_type OfficeHome/stylized-jigsaw --source product art realworld --target clipart --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sc_id --exp_type OfficeHome/stylized-jigsaw --source product art realworld --target clipart --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sc_id --exp_type OfficeHome/stylized-jigsaw --source product art realworld --target clipart --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sc_id --exp_type OfficeHome/stylized-jigsaw --source product art realworld --target clipart --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222


# Realworld Target: 1126

echo "Realworld"

python get_activations.py --run_id $sr_id --exp_type OfficeHome/stylized-jigsaw --source product art clipart --target realworld --generate_for product --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sr_id --exp_type OfficeHome/stylized-jigsaw --source product art clipart --target realworld --generate_for art --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sr_id --exp_type OfficeHome/stylized-jigsaw --source product art clipart --target realworld --generate_for clipart --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222

python get_activations.py --run_id $sr_id --exp_type OfficeHome/stylized-jigsaw --source product art clipart --target realworld --generate_for realworld --batch_size 128 --n_classes 65 --network resnet18 --jigsaw_n_classes 30 --image_size 222
