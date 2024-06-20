# Benchmarking_Everything

## 1. environment

Combine the two environments into a virtual environment in Benchmark

```shell
# for gaussian-splatting:
cd gaussian-splatting
conda env create --file environment.yml
conda activate benchmark
# for instant-ngp:
cd ../instant-ngp
cmake -DNGP_BUILD_WITH_GUI=off ./ -B ./build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j 16
pip install -r requirements.txt
cd ../
```

## 2. Download and generate dataset

for mitsuba3 dataset:

build docker first

```shell
cd Mitsuba3_material-main
chmod +x build_docker_mi.sh 
./build_docker_mi.sh
docker run -it --gpus all --name Mitsuba_material -v  <location-of-Benchmarking_Everything>:/home -d --restart="on-failure" --shm-size 6G -e NVIDIA_DRIVER_CAPABILITIES=all christy/mitsuba:stable
docker exec -it <your-container-id> /bin/bash
cd /home/Mitsuba3_material-main
conda activate mitsuba_material
```

In "Mitsuba3_material-main" floder:
generate material:

```shell
python resource/data/ior_to_file.py
```

Download and generate dataset by script:

```shell
chmod +x gen_images_w_mask.sh
./gen_images_w_mask.sh 0 ./scene/object  ../datasets/mitsuba3_synthetic/ 16
cd ../
```

for sphere:
```shell
chmod +x gen_sphere.sh
./gen_sphere.sh 0 ./example/sphere  ../datasets/sphere/ 16
cd ../
```

Since the above format image is not convenient to see the effect of the background, you can run the following code to generate preview images:

```shell
chmod +x gen_preview.sh
./gen_preview.sh 0 ./scene/object false
```

or for sphere:

```shell
chmod +x gen_preview.sh
./gen_preview.sh 0 ./example/sphere true
```

## 3. start training

for ${method}: method can be instant-nsr-pl and NeuS2

```shell
cd ${method}
chmod +x run_mitsuba.sh
bash run_mitsuba.sh $gpu 
# for example: bash run_mitsuba.sh 0
cd ../
``` 

the result of nerf are stored in the "instant-nsr-pl-output-womask/output.txt" in the following format:

```shell
${object}:${method}:${material}:${PSNR}-${SSIM}
```

## 4. Eval

eval after training with all methods

```shell
cd Mitsuba3_material-main
chmod +x eval/eval_mitsuba.sh
bash eval/eval_mitsuba.sh ../Mesh ../output
cd ../
```

for sphere:
```shell
cd Mitsuba3_material-main
chmod +x eval/eval_mitsuba.sh
bash eval/eval_mitsuba.sh ../Sphere_mesh ../sphere
cd ../
```

the result are stored in the "mesh_evaluation/output.txt" in the following format:

```shell
${object}:${method}:${material}:${cds}
```

## 5. Visualization

```shell
python pick.py
```

for neus2 and instant-nsr-pl-neus:

single-view and single-picture:

```shell
CUDA_VISIBLE_DEVICES=0 python eval/visualization.py --floor --picked --input_dir ../picked_mesh --pic_num 1
```

single-view and multi-pictures:

```shell
CUDA_VISIBLE_DEVICES=0 python eval/visualization.py --floor --picked --input_dir ../picked_mesh --start 0 --end 7
```

multi-views:

```shell
CUDA_VISIBLE_DEVICES=0 python eval/visualization.py --floor --picked --input_dir ../picked_mesh --start <start-number> --end <end-number>
```

for gaussian_splatting and instant-nsr-pl-nerf

```shell
python vis.py --pic_num 1
```
