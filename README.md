# Benchmarking_Everything

Due to the large dataset, we would like to use huggingface in conjunction with webdataset to achieve the goal of being able to use the data without having to download all of it, but at the moment that part of the code has not yet been sorted out.

## 1. start training

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

## 2. Eval

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
