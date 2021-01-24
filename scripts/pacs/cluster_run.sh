# printf "Clustering on source domains ...\n\n"

# printf "Vanilla\n\n"

# python cluster_eval.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 3811 --source_clustering

# python cluster_eval.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 6556 --source_clustering

# python cluster_eval.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 8913 --source_clustering

# python cluster_eval.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 1126 --source_clustering

# printf "\n\nStylized\n\n"

# python cluster_eval.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 3361 --source_clustering

# python cluster_eval.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 5584 --source_clustering

# python cluster_eval.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 8057 --source_clustering

# python cluster_eval.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 425 --source_clustering


#### Vanilla

vp_id=8369 # 0.01
va_id=1476 # 0.01
vc_id=3653 # 0.00001
vs_id=8400 # 0.01

#### Stylized
sp_id=9529 # 0.01
sa_id=2118 # 0.01
sc_id=4972 # 0.01
ss_id=8414 # 0.01

vanilla_folder_name="PACS/scratch-vanilla-jigsaw"
stylized_folder_name="PACS/scratch-stylized-jigsaw"

# printf "Clustering on target domains ...\n"

# printf "Vanilla\n"

# python cluster_eval.py --seed 1 --exp_type $vanilla_folder_name --source art cartoon sketch --target photo --run_id $vp_id --target_clustering --num_classes 7

# python cluster_eval.py --seed 1 --exp_type $vanilla_folder_name --source sketch cartoon photo --target art --run_id $va_id --target_clustering --num_classes 7

# python cluster_eval.py --seed 1 --exp_type $vanilla_folder_name --source art sketch photo --target cartoon --run_id $vc_id --target_clustering --num_classes 7

# python cluster_eval.py --seed 1 --exp_type $vanilla_folder_name --source art cartoon photo --target sketch --run_id $vs_id --target_clustering --num_classes 7


# printf "Stylized\n"

# python cluster_eval.py --seed 1 --exp_type $stylized_folder_name --source art cartoon sketch --target photo --run_id $sp_id --target_clustering --num_classes 7

# python cluster_eval.py --seed 1 --exp_type $stylized_folder_name --source sketch cartoon photo --target art --run_id $sa_id --target_clustering --num_classes 7

# python cluster_eval.py --seed 1 --exp_type $stylized_folder_name --source art sketch photo --target cartoon --run_id $sc_id --target_clustering --num_classes 7

# python cluster_eval.py --seed 1 --exp_type $stylized_folder_name --source art cartoon photo --target sketch --run_id $ss_id --target_clustering --num_classes 7


printf "Clustering on source domains ...\n"

printf "Vanilla\n"

python cluster_eval.py --seed 1 --exp_type $vanilla_folder_name --source art cartoon sketch --target photo --run_id $vp_id --source_clustering --num_classes 7

python cluster_eval.py --seed 1 --exp_type $vanilla_folder_name --source sketch cartoon photo --target art --run_id $va_id --source_clustering --num_classes 7

python cluster_eval.py --seed 1 --exp_type $vanilla_folder_name --source art sketch photo --target cartoon --run_id $vc_id --source_clustering --num_classes 7

python cluster_eval.py --seed 1 --exp_type $vanilla_folder_name --source art cartoon photo --target sketch --run_id $vs_id --source_clustering --num_classes 7


printf "Stylized\n"

python cluster_eval.py --seed 1 --exp_type $stylized_folder_name --source art cartoon sketch --target photo --run_id $sp_id --source_clustering --num_classes 7

python cluster_eval.py --seed 1 --exp_type $stylized_folder_name --source sketch cartoon photo --target art --run_id $sa_id --source_clustering --num_classes 7

python cluster_eval.py --seed 1 --exp_type $stylized_folder_name --source art sketch photo --target cartoon --run_id $sc_id --source_clustering --num_classes 7

python cluster_eval.py --seed 1 --exp_type $stylized_folder_name --source art cartoon photo --target sketch --run_id $ss_id --source_clustering --num_classes 7

