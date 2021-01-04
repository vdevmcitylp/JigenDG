
### Vanilla

vp_id=8276
va_id=8722
vc_id=9666
vr_id=941

folder_name="OfficeHome/DG"

echo "Linear Evalulation: Stylized Jigsaw"

echo "Product"

python LinearEval.py --seed 1 --run_id $vp_id --exp_type $folder_name --n_classes 65 --source art clipart realworld --target product --calc_for product

echo "Art"

python LinearEval.py --seed 1 --run_id $va_id --exp_type $folder_name --n_classes 65 --source product clipart realworld --target art --calc_for art

echo "Clipart"

python LinearEval.py --seed 1 --run_id $vc_id --exp_type $folder_name --n_classes 65 --source product art realworld --target clipart --calc_for clipart

echo "Realworld"

python LinearEval.py --seed 1 --run_id $vr_id --exp_type $folder_name --n_classes 65 --source product art clipart --target realworld --calc_for realworld


# echo "Linear Evalulation: Stylized Jigsaw"

# echo "Product"

# python LinearEval.py --seed 1 --run_id 6667 --exp_type OfficeHome/stylized-jigsaw --n_classes 65 --source art clipart realworld --target product 

# echo "Art"

# python LinearEval.py --seed 1 --run_id 1385 --exp_type OfficeHome/stylized-jigsaw --n_classes 65 --source product clipart realworld --target art 

# echo "Clipart"

# python LinearEval.py --seed 1 --run_id 6451 --exp_type OfficeHome/stylized-jigsaw --n_classes 65 --source product art realworld --target clipart 

# echo "Realworld"

# python LinearEval.py --seed 1 --run_id 1068 --exp_type OfficeHome/stylized-jigsaw --n_classes 65 --source product art clipart --target realworld

