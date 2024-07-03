# Benchmarking_Everything

Due to the large dataset, we would like to use huggingface in conjunction with webdataset to achieve the goal of being able to use the data without having to download all of it, but at the moment that part of the code has not yet been sorted out.

## 1. start training

for ${method}: method can be instant-nsr-pl and NeuS2

```shell
cd ${method}
chmod +x run_openmaterial.sh
bash run_openmaterial.sh $gpu 
# for example: bash run_openmaterial.sh 0
cd ../
``` 

the result of nerf are stored in the "instant-nsr-pl-output-womask/output.txt" in the following format:

```shell
${object}:${method}:${material}:${PSNR}-${SSIM}
```

## 2. Eval

eval after training with all methods

```shell
cd Openmaterial-main
chmod +x eval/eval.sh
bash eval/eval.sh ../Mesh ../output
cd ../
```


the result are stored in the "mesh_evaluation/output.txt" in the following format:

```shell
${object}:${method}:${material}:${cds}
```
