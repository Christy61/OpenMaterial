import os
import json
import pandas as pd
import re
import argparse
import glob
from tqdm import tqdm

bsdf_names = [
    'conductor',
    'dielectric',
    'plastic',
    'roughconductor',
    'roughdielectric',
    'roughplastic',
    'diffuse'
]

hdr_names = [
    'cobblestone_street_night_4k',
    'leadenhall_market_4k',
    'symmetrical_garden_4k'
    ]

object_names = {
    'b14a251fe8ad4a10bbc75f7dd3f6cebb': "vase",
    'fc4f34dae22c4dae95c19b1654c3cb7e': "snail",
    '01098ad7973647a9b558f41d2ebc5193': "Boat",
    'df894b66e2e54b558a43497becb94ff0': "Bike",
    '5c230ea126b943b8bc1da3f5865d5cd2': "Statue",
}

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='output-ablation')
parser.add_argument('--method', type=str,)
parser.add_argument('--eval_mesh', action='store_true')
args = parser.parse_args()

def sum_instant_nsr_pl():
    count_dir_bsdf, psnr_dir_bsdf, ssim_dir_bsdf, lpips_dir_bsdf = {}, {}, {}, {}
    count_dir_hdr, psnr_dir_hdr, ssim_dir_hdr, lpips_dir_hdr = {}, {}, {}, {}
    count_dir_obj, psnr_dir_obj, ssim_dir_obj, lpips_dir_obj = {}, {}, {}, {}
    folder_path = args.output_dir
    for name in bsdf_names:
        count_dir_bsdf[name] = 0
        psnr_dir_bsdf[name] = 0.0
        ssim_dir_bsdf[name] = 0.0
        lpips_dir_bsdf[name] = 0.0

    for name in hdr_names:
        count_dir_hdr[name] = 0
        psnr_dir_hdr[name] = 0.0
        ssim_dir_hdr[name] = 0.0
        lpips_dir_hdr[name] = 0.0

    for name in object_names.keys():
        count_dir_obj[object_names[name]] = 0
        psnr_dir_obj[object_names[name]] = 0.0
        ssim_dir_obj[object_names[name]] = 0.0
        lpips_dir_obj[object_names[name]] = 0.0

    for file_name in tqdm(object_names.keys(), desc="sum instant-nsr-pl..."):
        scene_names_ = os.listdir(os.path.join(folder_path, file_name))
        for scene_name in scene_names_:
            if scene_name.endswith("insr"):
                txt_path = os.path.join(folder_path, file_name, scene_name, 'instant-nsr-pl-wmask.txt')

                with open(txt_path, 'r') as f:
                    txt_data = f.readline()
                    psnr_ = re.search(r'PSNR=(\d+\.\d+)', txt_data).group(1)
                    ssim_ = re.search(r'SSIM=(\d+\.\d+)', txt_data).group(1)
                    lpips_ = re.search(r'lpips=(\d+\.\d+)', txt_data).group(1)
                    bsdf_name = txt_data.split(':')[2]

                    for name in bsdf_names:
                        if bsdf_name.startswith(name):
                            psnr_dir_bsdf[name] += float(psnr_)
                            ssim_dir_bsdf[name] += float(ssim_)
                            lpips_dir_bsdf[name] += float(lpips_)
                            count_dir_bsdf[name] += 1

                    for name in hdr_names:
                        if scene_name.startswith(name):
                            psnr_dir_hdr[name] += float(psnr_)
                            ssim_dir_hdr[name] += float(ssim_)
                            lpips_dir_hdr[name] += float(lpips_)
                            count_dir_hdr[name] += 1

                    for name in object_names.keys():
                        if file_name.startswith(name):
                            psnr_dir_obj[object_names[name]] += float(psnr_)
                            ssim_dir_obj[object_names[name]] += float(ssim_)
                            lpips_dir_obj[object_names[name]] += float(lpips_)
                            count_dir_obj[object_names[name]] += 1

    for name in bsdf_names:
        print(f"[+] {name} result: {count_dir_bsdf[name]}")
        if count_dir_bsdf[name] > 0:
            psnr_dir_bsdf[name] = psnr_dir_bsdf[name] / count_dir_bsdf[name]
            ssim_dir_bsdf[name] = ssim_dir_bsdf[name] / count_dir_bsdf[name]
            lpips_dir_bsdf[name] = lpips_dir_bsdf[name] / count_dir_bsdf[name]
    for name in hdr_names:
        print(f"[+] {name} result: {count_dir_hdr[name]}")
        if count_dir_hdr[name] > 0:
            psnr_dir_hdr[name] = psnr_dir_hdr[name] / count_dir_hdr[name]
            ssim_dir_hdr[name] = ssim_dir_hdr[name] / count_dir_hdr[name]
            lpips_dir_hdr[name] = lpips_dir_hdr[name] / count_dir_hdr[name]
    for name in object_names.keys():
        print(f"[+] {name} result: {count_dir_obj[object_names[name]]}")
        if count_dir_obj[object_names[name]] > 0:
            psnr_dir_obj[object_names[name]] = psnr_dir_obj[object_names[name]] / count_dir_obj[object_names[name]]
            ssim_dir_obj[object_names[name]] = ssim_dir_obj[object_names[name]] / count_dir_obj[object_names[name]]
            lpips_dir_obj[object_names[name]] = lpips_dir_obj[object_names[name]] / count_dir_obj[object_names[name]]
    return psnr_dir_bsdf, ssim_dir_bsdf, lpips_dir_bsdf, psnr_dir_hdr, ssim_dir_hdr, lpips_dir_hdr, psnr_dir_obj, ssim_dir_obj, lpips_dir_obj


def mesh_cds(method):
    cds_dir_bsdf, count_dir_bsdf = {}, {}
    cds_dir_hdr, count_dir_hdr = {}, {}
    cds_dir_obj, count_dir_obj = {}, {}
    folder_path = args.output_dir
    cds_dir_bsdf[method] = {}
    count_dir_bsdf[method] = {}
    cds_dir_hdr[method] = {}
    count_dir_hdr[method] = {}
    cds_dir_obj[method] = {}
    count_dir_obj[method] = {}

    for name in bsdf_names:
        count_dir_bsdf[method][name] = 0
        cds_dir_bsdf[method][name] = 0.0

    for name in hdr_names:
        count_dir_hdr[method][name] = 0
        cds_dir_hdr[method][name] = 0.0
    
    for name in object_names.keys():
        count_dir_obj[method][object_names[name]] = 0
        cds_dir_obj[method][object_names[name]] = 0.0

    for file_name in tqdm(object_names.keys(), desc="sum chamfer distance..."):
        file_name = file_name.strip()
        txt_paths = glob.glob(os.path.join(folder_path, file_name, '*mesh-output.txt'))
        for txt_path in txt_paths:
            with open(txt_path, 'r') as f:
                for line in f:
                    txt_data = line.strip()
                    method_name = txt_data.split(':')[1]
                    bsdf_name = txt_data.split(':')[2]
                    cds = txt_data.split(':')[-1]
                    if method_name == "instant-nsr-pl-wmask":
                        method_name = "insr"
                    if method_name == method:
                        for name in bsdf_names:
                            if bsdf_name.endswith(name):
                                cds_dir_bsdf[method][name] += float(cds)
                                count_dir_bsdf[method][name] += 1

                        for name in hdr_names:
                            if bsdf_name.startswith(name):
                                cds_dir_hdr[method][name] += float(cds)
                                count_dir_hdr[method][name] += 1

                        for name in object_names.keys():
                            if file_name.startswith(name):
                                cds_dir_obj[method][object_names[name]] += float(cds)
                                count_dir_obj[method][object_names[name]] += 1

    for name in bsdf_names:
        if count_dir_bsdf[method][name] > 0:
            cds_dir_bsdf[method][name] = cds_dir_bsdf[method][name] / count_dir_bsdf[method][name]
        else:
            cds_dir_bsdf[method][name] = 0.0
            print(f"Warning: No material type: {name}")
    for name in hdr_names:
        if count_dir_hdr[method][name] > 0:
            cds_dir_hdr[method][name] = cds_dir_hdr[method][name] / count_dir_hdr[method][name]
    for name in object_names.keys():
        if count_dir_obj[method][object_names[name]] > 0:
            cds_dir_obj[method][object_names[name]] = cds_dir_obj[method][object_names[name]] / count_dir_obj[method][object_names[name]]
    return cds_dir_bsdf, cds_dir_hdr, cds_dir_obj


if __name__ == '__main__':

    result = {}
    cds = {}
    flag = True

    method_list = [args.method]
    if args.eval_mesh:
        method_cds = [args.method]
    else:
        method_cds = []

    for method in method_list:
        if method == "insr":
            result['insr'] = sum_instant_nsr_pl()
            flag = True
    
    for method in method_cds:
        if method in method_cds:
            cds[method] = mesh_cds(method)

    if args.method == None or flag:
        print('\n')
        print("******************************************")
        print("           Novel View Synthesis           ")
        print("******************************************")

        print("--------------Material Type--------------")
        rows = method_list
        columns = [name for name in bsdf_names]
        PSNR_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                PSNR_df.at[r, c] = result[r][0][c]
        print("PSNR:")
        print(PSNR_df)
        SSIM_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                SSIM_df.at[r, c] = result[r][1][c]
        print("SSIM:")
        print(SSIM_df)
        LPIPS_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                LPIPS_df.at[r, c] = result[r][2][c]
        print("LPIPS:")
        print(LPIPS_df)
        print('\n')

        print("--------------Lighting Type--------------")
        rows = method_list
        columns = [name for name in hdr_names]
        PSNR_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                PSNR_df.at[r, c] = result[r][3][c]
        print("PSNR:")
        print(PSNR_df)
        SSIM_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                SSIM_df.at[r, c] = result[r][4][c]
        print("SSIM:")
        print(SSIM_df)
        LPIPS_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                LPIPS_df.at[r, c] = result[r][5][c]
        print("LPIPS:")
        print(LPIPS_df)
        print('\n')

        print("--------------Object Name--------------")
        rows = method_list
        columns = [name for name in object_names.values()]
        PSNR_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                PSNR_df.at[r, c] = result[r][6][c]
        print("PSNR:")
        print(PSNR_df)
        SSIM_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                SSIM_df.at[r, c] = result[r][7][c]
        print("SSIM:")
        print(SSIM_df)
        LPIPS_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                LPIPS_df.at[r, c] = result[r][8][c]
        print("LPIPS:")
        print(LPIPS_df)
        print('\n')

    if args.eval_mesh or args.method == None:
        print("*******************************************")
        print("             3D Reconstruction             ")
        print("*******************************************")

        print("--------------Material Type--------------")
        rows = method_cds
        columns = [name for name in bsdf_names]
        CDs_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                CDs_df.at[r, c] = cds[r][0][r][c]
        print("Chamfer Distance:")
        print(CDs_df)
        print('\n')

        print("--------------Lighting Type--------------")
        rows = method_cds
        columns = [name for name in hdr_names]
        CDs_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                CDs_df.at[r, c] = cds[r][1][r][c]
        print("Chamfer Distance:")
        print(CDs_df)
        print('\n')

        print("--------------Object Name--------------")
        rows = method_cds
        columns = [name for name in object_names.values()]
        CDs_df = pd.DataFrame(index=rows, columns=columns)
        for r in rows:
            for c in columns:
                CDs_df.at[r, c] = cds[r][2][r][c]
        print("Chamfer Distance:")
        print(CDs_df)
        print('\n')
