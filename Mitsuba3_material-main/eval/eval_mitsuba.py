# copy from: https://github.com/liuyuan-pal/NeRO/blob/main/eval_real_shape.py
import torch
import argparse
from pathlib import Path
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm
import os

def load_mesh(file_path):
    mesh = trimesh.load(file_path, process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
    faces = torch.tensor(mesh.faces, dtype=torch.int64).cuda()
    return Meshes(verts=[verts], faces=[faces]) , len(verts)


def nearest_dist(pts0, pts1, batch_size=512):
    pn0, pn1 = pts0.shape[0], pts1.shape[0]
    dists = []
    for i in tqdm(range(0, pn0, batch_size), desc='evaluting...'):
        dist = torch.norm(pts0[i:i+batch_size,None,:] - pts1[None,:,:], dim=-1)
        dists.append(torch.min(dist,1)[0])
    dists = torch.cat(dists,0).cpu().numpy()
    return dists


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pr',type=str,)
    parser.add_argument('--gt',type=str,)
    parser.add_argument('--object',type=str,)
    parser.add_argument('--method',type=str,)
    parser.add_argument('--output',type=str,)
    args = parser.parse_args()
    max_dist = 0.15

    mesh_pr, len_pr = load_mesh(f'{args.pr}')
    mesh_gt, len_gt = load_mesh(f'{args.gt}')

    num_samples = 1000000

    pts_pr = sample_points_from_meshes(mesh_pr, num_samples=num_samples).squeeze()
    pts_gt = sample_points_from_meshes(mesh_gt, num_samples=num_samples).squeeze()

    bn = 512
    dist_gt = nearest_dist(pts_gt, pts_pr, bn)
    mean_gt = dist_gt[dist_gt < max_dist].mean()
    dist_pr = nearest_dist(pts_pr, pts_gt, bn)
    mean_pr = dist_pr[dist_pr < max_dist].mean()
    
    stem = Path(args.pr).stem
    print(stem)
    chamfer = (mean_gt + mean_pr) / 2 * 100
    results = f'{args.object}:{args.method}:{stem}:{chamfer:.5f}'
    os.makedirs(f'{args.output}/{args.object}', exist_ok=True)
    with open(os.path.join(args.output, args.object, f"{args.method}-mesh-output.txt"), "w") as file:
        file.write(results + '\n')
    print(results)
