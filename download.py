from huggingface_hub import login, list_repo_files, hf_hub_download, snapshot_download
import os
import argparse
import multiprocessing

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
    if args.type in bsdf_names:
        material_type = args.type
        LOCAL_DIR = f"datasets/openmaterial"
        os.makedirs(LOCAL_DIR, exist_ok=True)
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=f"{material_type}-000000.tar", ignore_patterns="depth*.tar",local_dir=LOCAL_DIR, token=args.token)
        if args.depth:
            snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=f"depth-{material_type}*.tar",local_dir=LOCAL_DIR, token=args.token)
    elif args.type == 'all':
        LOCAL_DIR = "datasets/openmaterial"
        os.makedirs(LOCAL_DIR, exist_ok=True)
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns="*.tar", ignore_patterns="depth-*.tar",local_dir=LOCAL_DIR, token=args.token)
        if args.depth:
            snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns="depth*.tar", local_dir=LOCAL_DIR, token=args.token)
    else:
        raise ValueError("There's no such material.")

    os.makedirs("./dataset", exist_ok=True)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=f"groundtruth.tar", ignore_patterns="depth*.tar",local_dir="groundtruth", token=args.token)
