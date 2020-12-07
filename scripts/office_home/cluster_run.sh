printf "Clustering on source domains ...\n\n"

printf "Vanilla\n\n"

python cluster_eval.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 3811 --source_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 6556 --source_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 8913 --source_clustering

python cluster_eval.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 1126 --source_clustering

printf "\n\nStylized\n\n"

python cluster_eval.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 3361 --source_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 5584 --source_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 8057 --source_clustering

python cluster_eval.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 425 --source_clustering

# printf "Clustering on target domains ...\n"

# printf "Vanilla\n"

# python cluster_eval.py --exp_type vanilla-jigsaw --source art cartoon photo --target sketch --run_id 3811 --target_clustering

# python cluster_eval.py --exp_type vanilla-jigsaw --source art sketch photo --target cartoon --run_id 6556 --target_clustering

# python cluster_eval.py --exp_type vanilla-jigsaw --source sketch cartoon photo --target art --run_id 8913 --target_clustering

# python cluster_eval.py --exp_type vanilla-jigsaw --source art cartoon sketch --target photo --run_id 1126 --target_clustering

# printf "Stylized\n"

# python cluster_eval.py --exp_type stylized-jigsaw --source art cartoon photo --target sketch --run_id 3361 --target_clustering

# python cluster_eval.py --exp_type stylized-jigsaw --source art sketch photo --target cartoon --run_id 5584 --target_clustering

# python cluster_eval.py --exp_type stylized-jigsaw --source sketch cartoon photo --target art --run_id 8057 --target_clustering

# python cluster_eval.py --exp_type stylized-jigsaw --source art cartoon sketch --target photo --run_id 425 --target_clustering