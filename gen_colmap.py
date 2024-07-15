import os
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from shutil import copytree
import json
import numpy as np
import cv2 as cv

"""
Part of the code is adapted from:
https://gist.github.com/Totoro97/05e0b6afef5e580464731ad4c69c7a41

"""

def gen_w2c(pose):
    
    pose[:3, :1] = -pose[:3, :1]
    pose[:3, 1:2] = -pose[:3, 1:2]  # Flip the x+ and y+ to align coordinate system

    R = pose[:3, :3].transpose()
    T = -R @ pose[:3, 3:]
    return R, T

# "cv.decomposeProjectionMatrix(P)" generates some errors in special cases
# Refer to: https://stackoverflow.com/questions/55814640/decomposeprojectionmatrix-gives-unexpected-result
# so neither pitch nor yaw can be zero.
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def gen_camera_intrinsic(width, height, fov_x, fov_y):
    fx = width / 2.0 / math.tan(fov_x / 180 * math.pi / 2.0)
    fy = height / 2.0 / math.tan(fov_y / 180 * math.pi / 2.0)
    return fx, fy


def main():
    LOCAL_PATH = "datasets/openmaterial"
    
    for path in os.listdir(LOCAL_PATH): 
        path2 = os.listdir(os.path.join(LOCAL_PATH, path))
        output_dir = os.path.join(LOCAL_PATH, path)
        images_folder = os.path.join(os.path.dirname(__file__), f'./{output_dir}')
        os.makedirs(images_folder, exist_ok=True)

        file_train_path = os.path.join(LOCAL_PATH, path, path2[0], 'transforms_train.json')
        file_test_path = os.path.join(LOCAL_PATH, path, path2[0], 'transforms_test.json')
        train_paths = os.path.join(LOCAL_PATH, path, path2[0], 'train')
        write_colmap(train_paths, output_dir, file_train_path, flag=True, is_train=True)
        test_paths = os.path.join(LOCAL_PATH, path, path2[0], 'test')
        write_colmap(test_paths, output_dir, file_test_path, flag=True, is_train=False)
        
def write_colmap(img_path, output_dir, json_path, flag, is_train):
    object_name = os.path.split(output_dir)[-1]
    width, height = 1600, 1200 
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    fov_x = 37.8492
    fov_y = 28.8415

    scale_mat = np.diag([1.0, 1.0, 1.0, 1.0])
    os.makedirs(os.path.join(img_path, 'colmap', 'manually', 'sparse'), exist_ok=True)
    os.makedirs(os.path.join(img_path, 'colmap', 'dense'), exist_ok=True)

    # transform to Colmap format 
    fx, fy = gen_camera_intrinsic(width, height, fov_x, fov_y)

    # use float64 to avoid loss of precision
    intrinsic = np.diag([fx, fy, 1.0, 1.0]).astype(np.float64)
    # The origin is in the center and not in the upper left corner of the image
    intrinsic[0, 2] = width / 2.0
    intrinsic[1, 2] = height / 2.0

    # manually construct a sparse model by creating a cameras.txt, points3D.txt, and images.txt
    # Follow the steps at https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses
    with open(os.path.join(img_path, 'colmap', 'manually', 'sparse', 'points3D.txt'), 'w') as f:
        pass
    
    with open(os.path.join(img_path, 'colmap', 'manually', 'sparse', 'cameras.txt'), 'w') as f_camera:
        f_camera.write('1 PINHOLE 1600 1200 {} {} {} {}\n'.format(intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]))
        
    camera_params = {}
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    copytree(os.path.join(img_path, 'images'), os.path.join(img_path, 'colmap', 'images'), dirs_exist_ok=True)
    with open(os.path.join(img_path, 'colmap', 'manually', 'sparse', 'images.txt'), 'w') as f:
        for i, frame in enumerate(data['frames']):
            flip_mat = np.array([
				[-1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, -1, 0],
				[0, 0, 0, 1]
			])
            cam_pose_ = np.matmul(frame['transform_matrix'], flip_mat)
            # P = K[R|t]
            cam_pose = np.array(cam_pose_)
            R, T = gen_w2c(cam_pose)
            w2c = np.concatenate([np.concatenate([R, T], 1), bottom], 0)
            world_mat = intrinsic @ w2c

            camera_params['world_mat_%d' % i] = world_mat
            # Since the object has been normalised into the unit sphere, scale_mat is the unit array.
            camera_params['scale_mat_%d' % i] = scale_mat
            camera_params['intrinsic_mat_%d' % i] = intrinsic
            camera_params['extrinsic_mat_%d' % i] = w2c
            # P = world_mat @ scale_mat
            # P = P[:3, :4]
            # _, c2w_ = load_K_Rt_from_P(None,P)
            # R = c2w_[:3, :3].transpose()
            # T = -R @ c2w_[:3, 3:]

            rot = Rot.from_matrix(R)
            rot = rot.as_quat()
            image_name = '{:03d}.png'.format(i)
            f.write('{} {} {} {} {} {} {} {} {} {}\n\n'.format(i + 1, rot[3], rot[0], rot[1], rot[2], T[0, 0], T[1, 0], T[2, 0], 1, image_name))
    np.savez(f"{img_path}/cameras_sphere.npz", **camera_params)
    if flag and is_train:
        os.makedirs(f"../groundtruth/{object_name}", exist_ok=True)
        np.savez(f"../groundtruth/{object_name}/cameras_sphere.npz", **camera_params)
        flag = False
    elif flag and not is_train:
        np.savez(f"../groundtruth/{object_name}/cameras_sphere_test.npz", **camera_params)
        flag = False


if __name__ == '__main__':
    main()
