import numpy as np
import cv2 as cv
import os
from glob import glob
import trimesh
import argparse
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from mitsuba import ScalarTransform4f as T
import numpy as np
import json
import math

"""
This code is adapted from:
https://github.com/xxlong0/SparseNeuS/blob/main/evaluation/clean_mesh.py

"""

def gen_w2c(pose):
    
    pose[:3, :1] = -pose[:3, :1]
    pose[:3, 1:2] = -pose[:3, 1:2]  # Flip the x+ and y+ to align coordinate system

    R = pose[:3, :3].transpose()
    T = -R @ pose[:3, 3:]
    return R, T

def gen_camera_intrinsic(width, height, fov_x, fov_y):
    fx = width / 2.0 / math.tan(fov_x / 180 * math.pi / 2.0)
    fy = height / 2.0 / math.tan(fov_y / 180 * math.pi / 2.0)
    return fx, fy

def clean_points_by_mask(points, scene_name, imgs_idx=None, minimal_vis=0, mask_dilated_size=11):
    json_path = glob(f'../datasets/openmaterial/{scene_name}/*/transforms_train.json')[0]
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    fov_x = 37.8492
    fov_y = 28.8415
    width, height = 1600, 1200 
    # transform to Colmap format 
    fx, fy = gen_camera_intrinsic(width, height, fov_x, fov_y)
    
    # use float64 to avoid loss of precision
    intrinsic = np.diag([fx, fy, 1.0, 1.0]).astype(np.float64)
    # The origin is in the center and not in the upper left corner of the image
    intrinsic[0, 2] = width / 2.0
    intrinsic[1, 2] = height / 2.0
    flip_mat = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    scale_mat = np.diag([1.0, 1.0, 1.0, 1.0])
    
    mask_lis = sorted(glob(f'../datasets/openmaterial/{scene_name}/*/train/mask/*.png'))
    n_images = len(mask_lis)
    inside_mask = np.zeros(len(points))

    if imgs_idx is None:
        imgs_idx = [i for i in range(n_images)]

    for i, frame in enumerate(data['frames']):
        cam_pose_ = np.matmul(frame['transform_matrix'], flip_mat)
        cam_pose = np.array(cam_pose_)
        R, T = gen_w2c(cam_pose)
        w2c = np.concatenate([np.concatenate([R, T], 1), bottom], 0)
        world_mat = intrinsic @ w2c

        P = world_mat
        P = P @ scale_mat
        P = P[:3, :4]
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1

        mask_image = cv.imread(mask_lis[i])
        kernel_size = mask_dilated_size
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] > 128)

        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        in_mask = (pts_image[:, 0] >= 0) * (pts_image[:, 0] <= 1600) * (pts_image[:, 1] >= 0) * (
                pts_image[:, 1] <= 1200) > 0
        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]

        curr_mask = curr_mask.astype(np.float32) * in_mask

        inside_mask += curr_mask

        if i > len(imgs_idx):
            break

    return inside_mask > minimal_vis


def clean_points_by_visualhull(points, scene_name, imgs_idx=None, minimal_vis=0, mask_dilated_size=11):
    json_path = glob(f'../datasets/openmaterial/{scene_name}/*/transforms_train.json')[0]
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    fov_x = 37.8492
    fov_y = 28.8415
    width, height = 1600, 1200 

    # transform to Colmap format 
    fx, fy = gen_camera_intrinsic(width, height, fov_x, fov_y)

    # use float64 to avoid loss of precision
    intrinsic = np.diag([fx, fy, 1.0, 1.0]).astype(np.float64)
    # The origin is in the center and not in the upper left corner of the image
    intrinsic[0, 2] = width / 2.0
    intrinsic[1, 2] = height / 2.0
    flip_mat = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    scale_mat = np.diag([1.0, 1.0, 1.0, 1.0])
    mask_lis = sorted(glob(f'../datasets/openmaterial/{scene_name}/*/train/mask/*.png'))
    n_images = len(mask_lis)
    outside_mask = np.zeros(len(points))
    if imgs_idx is None:
        imgs_idx = [i for i in range(n_images)]

    for i in imgs_idx:
        cam_pose_ = np.matmul(data['frames'][i]['transform_matrix'], flip_mat)
        cam_pose = np.array(cam_pose_)
        R, T = gen_w2c(cam_pose)
        w2c = np.concatenate([np.concatenate([R, T], 1), bottom], 0)
        world_mat = intrinsic @ w2c

        P = world_mat
        P = P @ scale_mat
        P = P[:3, :4]
        
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1

        mask_image = cv.imread(mask_lis[i])
        kernel_size = mask_dilated_size  # default 101
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] < 128)  # * outside the mask

        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        border = 50
        in_mask = (pts_image[:, 0] >= (0 + border)) * (pts_image[:, 0] <= (1600 - border)) * (
                pts_image[:, 1] >= (0 + border)) * (
                          pts_image[:, 1] <= (1200 - border)) > 0
        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]

        curr_mask = curr_mask.astype(np.float32) * in_mask

        outside_mask += curr_mask

    return outside_mask < 5, scale_mat

def find_closest_point(p1, d1, p2, d2):
    # Calculate the direction vectors of the lines
    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)

    # Create the coefficient matrix A and the constant vector b
    A = np.vstack((d1_norm, -d2_norm)).T
    b = p2 - p1

    # Solve the linear system to find the parameters t1 and t2
    t1, t2 = np.linalg.lstsq(A, b, rcond=None)[0]

    # Calculate the closest point on each line
    closest_point1 = p1 + d1_norm * t1
    closest_point2 = p2 + d2_norm * t2

    # Calculate the average of the two closest points
    closest_point = 0.5 * (closest_point1 + closest_point2)

    return closest_point

def clean_mesh_faces_by_mask(mesh_file, new_mesh_file, scene_name, imgs_idx, cut_y=-1.0, minimal_vis=0, mask_dilated_size=11):
    old_mesh = trimesh.load(mesh_file)
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]
    mask = clean_points_by_mask(old_vertices, scene_name, imgs_idx, minimal_vis, mask_dilated_size)
    y_mask = old_vertices[:, 1] >= cut_y
    mask = mask & y_mask
    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.int64)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)
    new_mesh.export(new_mesh_file)


def clean_mesh_faces_by_visualhull(mesh_file, new_mesh_file, scene_name, imgs_idx, minimal_vis=0, mask_dilated_size=11):
    old_mesh = trimesh.load(mesh_file)
    os.remove(mesh_file)
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]
    mask, scale = clean_points_by_visualhull(old_vertices, scene_name, imgs_idx, minimal_vis, mask_dilated_size)
    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.int64)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)
    new_mesh.vertices *= scale[0, 0]
    new_mesh.vertices += scale[:3, 3]
    # ! if colmap trim=7, comment these
    # meshes = new_mesh.split(only_watertight=False)
    # new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]

    new_mesh.export(new_mesh_file)

def load_object(scene_dict, file_name):
    object_dir = {
        'type': 'ply', 
        'id': 'Material_0001',
        'filename': file_name, 
        'to_world': T([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]),
    }
    scene_dict['shape_0'] = object_dir
    return scene_dict

def load_integrator(scene_dict):
    integrator_dir = {
        'type': 'path',
        'max_depth': 65
    }
    scene_dict['integrator'] = integrator_dir
    return scene_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--directory', type=str)
    parser.add_argument('--object_name', type=str)
    args = parser.parse_args()
    mask_kernel_size = 11
    directory=args.directory
    dir_list = os.listdir(f'{directory}/meshes')
    object_name = args.object_name

    scene_list = os.listdir(f'{directory}/meshes/{object_name}')
    for scene_name in scene_list:
        scene_name = scene_name.split(".")[0]
        base_path = f"{directory}/meshes/{object_name}"
        print("processing:", scene_name)
        old_mesh_file_list = glob(os.path.join(base_path, "*.obj")) + glob(os.path.join(base_path, "*.ply"))
        for old_mesh_file in old_mesh_file_list:
            clean_mesh_file = os.path.join(base_path, f"clean_{scene_name}.ply")
            os.makedirs("{}/CleanedMesh/{}".format(directory, object_name), exist_ok=True)
            visualhull_mesh_file = f'{directory}/CleanedMesh/{object_name}/{scene_name}.ply'
            scene_path = glob(os.path.join('../datasets/groundtruth', object_name, '*.ply'))[0]
            scene_path = os.path.abspath(scene_path)
            scene_dict= {'type': 'scene'}
            scene_dict = load_integrator(scene_dict)
            scene_dict = load_object(scene_dict, scene_path)
            scene = mi.load_dict(scene_dict)
            bbox = scene.bbox()
            cut_y = bbox.min.y
            
            clean_mesh_faces_by_mask(old_mesh_file, clean_mesh_file, object_name, None, cut_y=cut_y, minimal_vis=2,
                                    mask_dilated_size=mask_kernel_size)

            clean_mesh_faces_by_visualhull(clean_mesh_file, visualhull_mesh_file, object_name, None, minimal_vis=2,
                                        mask_dilated_size=mask_kernel_size + 20)

            print("finish processing ", scene_name)