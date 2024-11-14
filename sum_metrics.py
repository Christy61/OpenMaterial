import os
import json
import pandas as pd
import re
import argparse
import glob

bsdf_names = [
    'diffuse',
    'dielectric',
    'roughdielectric',
    'conductor',
    'roughconductor',
    'plastic',
    'roughplastic'
]

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='output')
args = parser.parse_args()

def sum_instant_nsr_pl():
    psnr_dir, ssim_dir, lpips_dir, count_dir = {}, {}, {}, {}
    folder_path = args.output_dir
    object_names = os.listdir(folder_path)
    for name in bsdf_names:
        count_dir[name] = 0
        psnr_dir[name] = 0.0
        ssim_dir[name] = 0.0
        lpips_dir[name] = 0.0

    for file_name in object_names:
        txt_path = os.path.join(folder_path, file_name, 'instant-nsr-pl-wmask.txt')

        with open(txt_path, 'r') as f:
            txt_data = f.read()
            psnr_ = re.search(r'PSNR=(\d+\.\d+)', txt_data).group(1)
            ssim_ = re.search(r'SSIM=(\d+\.\d+)', txt_data).group(1)
            lpips_ = re.search(r'lpips=(\d+\.\d+)', txt_data).group(1)
            method = txt_data.split(':')[1]
            bsdf_name = txt_data.split(':')[2]

            for name in bsdf_names:
                if bsdf_name.startswith(name):
                    psnr_dir[name] += float(psnr_)
                    ssim_dir[name] += float(ssim_)
                    lpips_dir[name] += float(lpips_)
                    count_dir[name] += 1

            # print(psnr_dir)
            # print(ssim_dir)
            # print(lpips_dir)

    for name in bsdf_names:
        if count_dir[name] > 0:
            print(f"[+] {name} result: {count_dir[name]}")
            psnr_dir[name] = psnr_dir[name] / count_dir[name]
            ssim_dir[name] = ssim_dir[name] / count_dir[name]
            lpips_dir[name] = lpips_dir[name] / count_dir[name]
    return psnr_dir, ssim_dir, lpips_dir

def mesh_cds():
    cds_dir, count_dir = {}, {}
    folder_path = args.output_dir
    object_names = os.listdir(folder_path)
    count_dir['instant-nsr-pl-wmask'] = {}
    cds_dir['instant-nsr-pl-wmask'] = {}
    for name in bsdf_names:
        count_dir['instant-nsr-pl-wmask'][name] = 0
        cds_dir['instant-nsr-pl-wmask'][name] = 0.0

    for file_name in object_names:
        txt_paths = glob.glob(os.path.join(folder_path, file_name, '*mesh-output.txt'))
        print(txt_paths)
        for txt_path in txt_paths:
            with open(txt_path, 'r') as f:
                for line in f:
                    txt_data = line.strip()
                    method = txt_data.split(':')[1]
                    bsdf_name = txt_data.split(':')[2]
                    cds = txt_data.split(':')[-1]

                    for name in bsdf_names:
                        if bsdf_name.startswith(name):
                            cds_dir[method][name] += float(cds)
                            count_dir[method][name] += 1

    for name in bsdf_names:
        cds_dir['instant-nsr-pl-wmask'][name] = cds_dir['instant-nsr-pl-wmask'][name] / count_dir['instant-nsr-pl-wmask'][name]
    return cds_dir


if __name__ == '__main__':
    psnr, ssim, lpips = {}, {}, {}
    psnr['instant-nsr-pl(nerf-wmask)'], ssim['instant-nsr-pl(nerf-wmask)'], lpips['instant-nsr-pl(nerf-wmask)'] = sum_instant_nsr_pl()
    cds = mesh_cds()
    rows = ['instant-nsr-pl(nerf-wmask)']
    columns = [name for name in bsdf_names]
    PSNR_df = pd.DataFrame(index=rows, columns=columns)
    for r in rows:
        for c in columns:
            PSNR_df.at[r, c] = psnr[r][c]
    print("PSNR:")
    print(PSNR_df)
    print('\n')
    SSIM_df = pd.DataFrame(index=rows, columns=columns)
    for r in rows:
        for c in columns:
            SSIM_df.at[r, c] = ssim[r][c]
    print("SSIM:")
    print(SSIM_df)
    print('\n')
    LPIPS_df = pd.DataFrame(index=rows, columns=columns)
    for r in rows:
        for c in columns:
            LPIPS_df.at[r, c] = lpips[r][c]
    print("LPIPS:")
    print(LPIPS_df)
    print('\n')

    rows = ['instant-nsr-pl-wmask']
    columns = [name for name in bsdf_names]
    CDs_df = pd.DataFrame(index=rows, columns=columns)
    for r in rows:
        for c in columns:
            CDs_df.at[r, c] = cds[r][c]
    print("Chamfer Distance:")
    print(CDs_df)
