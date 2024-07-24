#!/bin/bash

directory=$1
output_dir=$2

if [ -d "$directory" ]; then
    # for subdir1 in "$directory"/*; do
        # subdir_name1=$(basename "$subdir1")
        subdir_name1='instant-nsr-pl-wmask'
        for subdir2 in "${directory}/${subdir_name1}/meshes/"*; do
            subdir_name2=$(basename "$subdir2")
            echo "clean_mesh start: ${subdir_name1}"
            python eval/clean_mesh.py --method ${subdir_name1} --directory ${directory}/${subdir_name1} --object_name ${subdir_name2}
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