#!/bin/bash

# Create separate txt_files for Stylized PACS experiments.

# data/
# 	txt_files/
# 		StylizedPACS/
# 			photo_target/
# 				photo_train.txt
# 				photo_test.txt
# 				art_train.txt
# 				art_test.txt
# 				...
#			art_target/
#				photo_train.txt
# 				photo_test.txt
# 				art_train.txt
# 				art_test.txt
# 				...

# Assumes the original author Github configuration, i.e. /home/fmc/data/PACS/kfold/photo/...

cd data/txt_lists

pacs_domains=( "photo" "art" "cartoon" "sketch" )
stylized_root="data/txt_lists/StylizedPACS"

# Change from /home/fmc/... to datasets root
dataset_root="/DATA1/vaasudev_narayanan/datasets"

for fname in *.txt 
do 
	sed -i "s@/home/fmc/data/@$dataset_root/@g" $fname
done

for domain in "${pacs_domains[@]}"
do
	mkdir -p "StylizedPACS/${domain}_target"
	echo $domain
	for fname in "photo_train.txt" "photo_test.txt" "art_train.txt" "art_test.txt" \
				"cartoon_train.txt" "cartoon_test.txt" "sketch_train.txt" "sketch_test.txt" 
	do 
		echo "StylizedPACS/${domain}_target/${fname}"
		sed "s@${dataset_root}/PACS/kfold/@${dataset_root}/StylizedPACS/${domain}_target/@g" $fname > "StylizedPACS/${domain}_target/${fname}"
	done

done
