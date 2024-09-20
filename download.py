from huggingface_hub import login, snapshot_download
import os
import argparse
from glob import glob

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
parser.add_argument('--token', type=str, help='your own token', required=True)
# Here you have to change to your own token
# 1. Click on your avatar in the upper right corner and select "Settings".
# 2. On the "Settings" page, click "Access Tokens" on the left side.
# 3. Generate a new Token and copy it.

parser.add_argument('--type', type=str, help='Types of materials to download, all for all downloads', required=True)
# Either choose one from bsdf_names or "all"

parser.add_argument('--depth', action='store_true', help='Whether depth data is required')
# Whether depth data is required

args = parser.parse_args()

login(token=args.token)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

REPO_ID = "EPFL-CVLab/OpenMaterial"

if __name__ == "__main__":    
    BASE_DIR = f"datasets"
    LOCAL_DIR = f"datasets/openmaterial"
    if args.type in bsdf_names:
        material_type = args.type
        os.makedirs(LOCAL_DIR, exist_ok=True)
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=f"{material_type}-*.tar", ignore_patterns=["depth-*.tar", "groundtruth.tar", "ablation-*.tar"],local_dir=LOCAL_DIR, token=args.token)
        tar_paths = glob(os.path.join(LOCAL_DIR, f"{material_type}-*.tar"))
        for tar_path in tar_paths:
            cmd = f'tar -xvf {tar_path} -C {LOCAL_DIR}'
            print(cmd)
            os.system(cmd)
        if args.depth:
            snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=f"depth-{material_type}*.tar",ignore_patterns="groundtruth.tar", local_dir=LOCAL_DIR, token=args.token)
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=f"groundtruth.tar", local_dir="./datasets", token=args.token)
        cmd = f'tar -xvf ./datasets/groundtruth.tar -C ./datasets'
        print(cmd)
        os.system(cmd)
    elif args.type == 'ablation':
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=f"ablation-*.tar", ignore_patterns=["depth-*.tar", "groundtruth.tar"],local_dir=LOCAL_DIR, token=args.token)
        tar_paths = glob(os.path.join(LOCAL_DIR, f"ablation-*.tar"))
        for tar_path in tar_paths:
            cmd = f'tar -xvf {tar_path} -C {LOCAL_DIR}'
            print(cmd)
            os.system(cmd)
        if args.depth:
            snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=f"depth-ablation*.tar", local_dir=LOCAL_DIR, token=args.token)
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=f"groundtruth-ablation.tar", local_dir="./datasets", token=args.token)
        cmd = f'tar -xvf ./datasets/groundtruth-ablation.tar -C ./datasets'
        print(cmd)
        os.system(cmd)
    elif args.type == 'all':
        os.makedirs(LOCAL_DIR, exist_ok=True)
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns="*.tar", ignore_patterns=["depth-*.tar", "groundtruth.tar", "ablation-*.tar", "groundtruth-ablation.tar"], local_dir=LOCAL_DIR, token=args.token)
        tar_paths = glob(os.path.join(LOCAL_DIR, "*.tar"))
        for tar_path in tar_paths:
            cmd = f'tar -xvf {tar_path} -C {LOCAL_DIR}'
            print(cmd)
            os.system(cmd)
        if args.depth:
            snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns="depth*.tar", local_dir=LOCAL_DIR, token=args.token)
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=f"groundtruth.tar", local_dir="./datasets", token=args.token)
        cmd = f'tar -xvf ./datasets/groundtruth.tar -C ./datasets'
        print(cmd)
        os.system(cmd)
    else:
        raise ValueError("There's no such material.")

    cmd = f'rm -r {BASE_DIR}/*.tar'
    print(cmd)
    os.system(cmd)

    cmd = f'rm -r {LOCAL_DIR}/*.tar'
    print(cmd)
    os.system(cmd)

    cmd = f'rm -r {LOCAL_DIR}/.cache'
    if os.path.exists(f"{LOCAL_DIR}/.cache"):
        print(cmd)
        os.system(cmd)

    cmd = f'rm -r {LOCAL_DIR}/.huggingface'
    if os.path.exists(f"{LOCAL_DIR}/.huggingface"):
        print(cmd)
        os.system(cmd)

    cmd = f'rm -r {BASE_DIR}/.cache'
    if os.path.exists(f"{BASE_DIR}/.cache"):
        print(cmd)
        os.system(cmd)

    cmd = f'rm -r {BASE_DIR}/.huggingface'
    if os.path.exists(f"{BASE_DIR}/.huggingface"):
        print(cmd)
        os.system(cmd)