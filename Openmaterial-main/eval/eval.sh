#!/bin/bash

directory=$1
output_dir=$2
method=$3
ablation=$4
if [ -d "$directory" ]; then
    subdir_name1=$method
    # for subdir1 in "$directory"/*; do
        # subdir_name1=$(basename "$subdir1")
        for subdir2 in "${directory}/${subdir_name1}/meshes/"*; do
            subdir_name2=$(basename "$subdir2")
            echo "clean_mesh start: ${subdir_name1}"
            if [ "$ablation" = "true" ]; then
                python eval/clean_mesh.py --dataset_dir ../datasets/ablation \
                --groundtruth_dir ../datasets/groundtruth_ablation \
                --method ${subdir_name1} \
                --directory ${directory}/${subdir_name1} \
                --object_name ${subdir_name2}
            else
                python eval/clean_mesh.py --method ${subdir_name1} --directory ${directory}/${subdir_name1} --object_name ${subdir_name2}
            fi
            echo "evaluation start:"
            for subdir3 in "${directory}/${subdir_name1}/CleanedMesh/${subdir_name2}/"*; do
                subdir_name3=$(basename "$subdir3")
                python eval/eval.py \
                --pr ${directory}/${subdir_name1}/CleanedMesh/${subdir_name2}/${subdir_name3} \
                --gt ../datasets/groundtruth/${subdir_name2}/clean_${subdir_name2}.ply \
                --object ${subdir_name2} \
                --method ${subdir_name1} \
                --output ${output_dir}
            done
        done
    # done
else
    echo "no dataset, please check again."
fi