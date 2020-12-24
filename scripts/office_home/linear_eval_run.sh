
echo "Linear Evalulation: Stylized Jigsaw"

echo "Product"

python LinearEval.py --seed 1 --run_id 2016 --exp_type OfficeHome/vanilla-jigsaw --n_classes 65 --source art clipart realworld --target product 

echo "Art"

python LinearEval.py --seed 1 --run_id 319 --exp_type OfficeHome/vanilla-jigsaw --n_classes 65 --source product clipart realworld --target art 

echo "Clipart"

python LinearEval.py --seed 1 --run_id 8174 --exp_type OfficeHome/vanilla-jigsaw --n_classes 65 --source product art realworld --target clipart 

echo "Realworld"

python LinearEval.py --seed 1 --run_id 5804 --exp_type OfficeHome/vanilla-jigsaw --n_classes 65 --source product art clipart --target realworld


# echo "Linear Evalulation: Stylized Jigsaw"

# echo "Product"

# python LinearEval.py --seed 1 --run_id 6667 --exp_type OfficeHome/stylized-jigsaw --n_classes 65 --source art clipart realworld --target product 

# echo "Art"

# python LinearEval.py --seed 1 --run_id 1385 --exp_type OfficeHome/stylized-jigsaw --n_classes 65 --source product clipart realworld --target art 

# echo "Clipart"

# python LinearEval.py --seed 1 --run_id 6451 --exp_type OfficeHome/stylized-jigsaw --n_classes 65 --source product art realworld --target clipart 

# echo "Realworld"

# python LinearEval.py --seed 1 --run_id 1068 --exp_type OfficeHome/stylized-jigsaw --n_classes 65 --source product art clipart --target realworld

