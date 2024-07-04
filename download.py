from huggingface_hub import login, list_repo_files, hf_hub_download
import os
import argparse


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

args = parser.parse_args()

login(token=args.token)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

REPO_ID = "EPFL-CVLab/OpenMaterial"

if args.type in bsdf_names:
    material_type = args.type
    LOCAL_DIR = f"datasets/openmaterial"
    os.makedirs(LOCAL_DIR, exist_ok=True)
    files_to_download = [f"{material_type}-{i:06d}.tar" for i in range(1024)]  # Adjust the range based on the number of shards
elif args.type == 'all':
    LOCAL_DIR = "datasets/openmaterial"
    os.makedirs(LOCAL_DIR, exist_ok=True)
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    files_to_download = [file for file in all_files if file.endswith('.tar')]
else:
    raise ValueError("There's no such material.")

# Download the dataset
for filename in files_to_download:
    try:
        local_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename, local_dir=LOCAL_DIR)
        print(f"Downloaded {filename} to {local_path}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
