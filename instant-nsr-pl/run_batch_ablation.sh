#!/bin/bash
start=${1:-0}
end=${2:-10}

root_dir="../datasets/ablation"

# rm -rf ../Mesh/instant-nsr-pl-wmask/

if [ -d "$root_dir" ]; then
    root_dir="${root_dir%/}"
    base_dir=$(basename "$root_dir")

    IFS=$'\n' read -d '' -r -a directories < <(find "$root_dir" -maxdepth 1 -type d | while IFS= read -r directory; do basename "$directory"; done | sort | grep -v "^$base_dir$" && printf '\0')
    # cut into groups
    echo "[+] Number of directories: ${#directories[@]}"
    declare -a group
    count=0
    for dir in "${directories[@]}"; do
        let count+=1
        if [ $count -gt $start ]; then
            group+=("$dir")
            if [ $count -eq $end ]; then
                echo "[+] from $start to $end, total count: ${#group[@]} in group"
                for g in "${group[@]}"; do
                    echo "group: $g"
                done
                break
            fi
        fi
    done

    for subdir1 in "${group[@]}"; do
        subdir_name1=$(basename "$subdir1")
        for subdir2 in "$root_dir/$subdir_name1"/*; do
          subdir_name2=$(basename "$subdir2")
        echo "[+] Case $subdir_name1"
        bsdf_name="${subdir_name2#*-}"

        python launch.py \
        --config configs/neus-openmaterial-wmask.yaml \
        --output_dir ../Mesh-ablation/instant-nsr-pl-wmask/meshes/${subdir_name1} \
        --gpu 0 \
        --train \
        dataset.bsdf_name=${bsdf_name} \
        dataset.object=${subdir_name1} \
        dataset.scene=${subdir_name2} \
        dataset.root_dir=${root_dir}/${subdir_name1}/${subdir_name2} \
        trial_name=${subdir_name1}
        rm -r exp/neus-openmaterial-wmask-${subdir_name2}/${subdir_name1}

        new_line="            with open(os.path.join(f'../output-ablation', self.config.dataset.object, f'{self.config.dataset.scene}+insr', \"instant-nsr-pl-wmask.txt\"), \"a\") as file:"
        sed -i "219s|.*|$new_line|"  systems/nerf.py
        CUDA_VISIBLE_DEVICES=1 python launch.py \
        --config configs/nerf-openmaterial-wmask.yaml \
        --gpu 0 \
        --train \
        dataset.bsdf_name=${bsdf_name} \
        dataset.object=${subdir_name1} \
        dataset.scene=${subdir_name2} \
        dataset.root_dir=${root_dir}/${subdir_name1}/${subdir_name2} \
        trial_name=${subdir_name1} \
        render_save_dir=../output-ablation
        rm -r exp/nerf-openmaterial-wmask-${subdir_name2}/${subdir_name1}

        done
    done
else
    echo "no openmaterial dataset, please generate before running."
fi
# rm -rf exp/