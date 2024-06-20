#!/bin/bash
directory="../datasets/openmaterial"

# rm -rf ../Mesh/instant-nsr-pl-wmask/

if [ -d "$directory" ]; then
    for subdir1 in "$directory"/*; do
        subdir_name1=$(basename "$subdir1")
        for subdir2 in "$directory/$subdir_name1"/*; do
          subdir_name2=$(basename "$subdir2")
        bsdf_name="${subdir_name2#*-}"

        python launch.py \
        --config configs/neus-openmaterial-wmask.yaml \
        --output_dir ../Mesh/instant-nsr-pl-wmask/meshes/${subdir_name1} \
        --gpu $1 \
        --train \
        dataset.bsdf_name=${bsdf_name} \
        dataset.object=${subdir_name1} \
        dataset.scene=${subdir_name2} \
        dataset.root_dir=${directory}/${subdir_name1}/${subdir_name2} \
        trial_name=${subdir_name1}
        # rm -r exp/neus-openmaterial-wmask-${subdir_name2}/${subdir_name1}

        # python launch.py \
        # --config configs/neus-openmaterial-womask.yaml \
        # --output_dir ../Mesh/instant-nsr-pl-womask/meshes/${subdir_name1} \
        # --gpu $1 \
        # --train \
        # dataset.bsdf_name=${bsdf_name} \
        # dataset.object=${subdir_name1} \
        # dataset.scene=${subdir_name2} \
        # dataset.root_dir=${directory}/${subdir_name1}/${subdir_name2} \
        # trial_name=${subdir_name1}

        new_line="            with open(os.path.join(f'../output', self.config.dataset.object, \"instant-nsr-pl-wmask.txt\"), \"w\") as file:"
        sed -i "219s|.*|$new_line|"  systems/nerf.py
        python launch.py \
        --config configs/nerf-openmaterial-wmask.yaml \
        --gpu $1 \
        --train \
        dataset.bsdf_name=${bsdf_name} \
        dataset.object=${subdir_name1} \
        dataset.scene=${subdir_name2} \
        dataset.root_dir=${directory}/${subdir_name1}/${subdir_name2} \
        trial_name=${subdir_name1} \
        render_save_dir=../output
        rm -r exp/nerf-openmaterial-wmask-${subdir_name2}/${subdir_name1}
        
        # new_line="            with open(os.path.join(f'../output', self.config.dataset.object, \"instant-nsr-pl-womask.txt\"), \"w\") as file:"
        # sed -i "219s|.*|$new_line|"  systems/nerf.py
        # python launch.py \
        # --config configs/nerf-openmaterial-womask.yaml \
        # --gpu $1 \
        # --train \
        # dataset.bsdf_name=${bsdf_name} \
        # dataset.object=${subdir_name1} \
        # dataset.scene=${subdir_name2} \
        # dataset.root_dir=${directory}/${subdir_name1}/${subdir_name2} \
        # trial_name=${subdir_name1} \
        # render_save_dir=../output

        done
    done
else
    echo "no openmaterial dataset, please generate before running."
fi
# rm -rf exp/