# Benchmarking_Everything

## 1. download dataset

Get your own token
1. Click on your avatar in the upper right corner and select "Settings".
2. On the "Settings" page, click "Access Tokens" on the left side.
3. Generate a new Token and copy it.

```shell
python download.py --token <your-token> --type all
``` 

unzip files
```shell
# for example
tar -xvf diffuse-*.tar
rm -r diffuse-*.tar
``` 

an example for using depth data (Here are the real depth values, not normalised):

```python
with h5py.File(filename, 'r') as hdf:
    dataset = hdf['depth']
    depth = dataset[:]  # size: (1200, 1600) 
```

## 2. start training

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

## 3. Eval

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

Run the following command to integrate the results:

```shell
python sum_metrics.py
```