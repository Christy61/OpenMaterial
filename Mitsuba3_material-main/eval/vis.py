import trimesh
import os
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from mitsuba import ScalarTransform4f as T
import numpy as np
from PIL import Image
import argparse
import cv2 as cv
import glob
import pymeshlab
from tqdm import tqdm

class ImageGrid:
    def pil_to_np(self, img):
        return np.array(img)
    
    def np_to_pil(self, img_np):
        return Image.fromarray(img_np)
    
    def get_rgb_image_(self, img):
        return self.pil_to_np(img)
    
    def get_image_grid(self, imgs_2d):
        rows = [np.concatenate([self.get_rgb_image_(img) for img in row_imgs], axis=1) for row_imgs in imgs_2d]
        grid = np.concatenate(rows, axis=0)
        return self.np_to_pil(grid)

def load_floor(scene_dict, bbox):
    trans = np.array([[-1, 1, 0, 0], 
            [0, 0, 1.414, bbox.min.y],
            [-1, -1, 0, 0],
            [0, 0, 0, 1]])
    scale_mat = [
        [20, 0, 0, 0], 
        [0, 20, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    to_world = trans @ scale_mat
    floor_dir = {
        "type": "rectangle",
        "id": "Floor",
        "to_world": T(to_world),
        "bsdf": {
            'type': 'diffuse', 
            'reflectance': {
                'type': 'rgb', 
                'value': [0.9, 0.9, 0.9]
            }
        }
    }
    scene_dict['shape_floor'] = floor_dir
    return scene_dict


def load_sensor(cam_to_world=None, spp=256):    
    if cam_to_world == None:
        cam_to_world = mi.ScalarTransform4f.look_at(
                        origin = [20.0, 3.09855, 0.0],
                        target = [0.0, 3.09855, 0.0],
                        up = [0, 1, 0])
        
    return mi.load_dict({
        'type': 'perspective',
        'fov': 28.8415,
        'fov_axis': 'smaller',
        'focus_distance': 6.0,
        'to_world':cam_to_world,
        'sampler': {
            'type': 'independent',
            'sample_count': spp
        },
        'film': {
            'type': 'hdrfilm',
            'width': 1600,
            'height': 1200,
            'rfilter': {
                # must be box for using denoiser 
                'type': 'box',  
            },
            'pixel_format': 'rgb',
        },
    })


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


def load_object(scene_dict, file_name, flip=False):
    object_dir = {
        'type': 'ply', 
        'id': 'Material_0001',
        'filename': file_name, 
        'flip_normals': flip,
        'to_world': T([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]), 
        'bsdf': {
            'type': 'twosided',
            'material':{
            'type': 'diffuse', 
            "reflectance": {
                "type": "rgb",
                "value": [0.38, 0.77, 0.88]
            }
            }
        }
    }
    scene_dict['shape_0'] = object_dir
    return scene_dict


def load_emitter(scene_dict, pose):
    angle = np.pi/3
    pitch = np.pi/2
    x = 4.2 * np.cos(angle) * np.cos(pitch)
    y = 4.2 * np.sin(pitch)
    z = 4.2 * np.sin(angle) * np.cos(pitch)
    trans = [[0, 0, -20, x],
        [-0, 20, 0, y],
        [20, 0, 0, z],
        [0, 0, 0, 1]]
    to_world = pose @ trans
    to_world = T(to_world)
    emitter_dir = {
        'type': 'envmap',
        'filename': f'textures/envmap.hdr'
    }
    scene_dict['shape_emitter'] = emitter_dir
    return scene_dict


def load_integrator(scene_dict):
    integrator_dir = {
        'type': 'path',
        'max_depth': 65,
        'hide_emitters': True
    }
    scene_dict['integrator'] = integrator_dir
    return scene_dict

def recalculate_vertex_normals(mesh_file, object_name):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file)
    ms.compute_normal_per_vertex()
    # The sub-implementation helps to further correct other normals 
    # on top of the neighbouring normals that have already been corrected.
    os.makedirs(f'../visualization/tmp/', exist_ok=True)
    new_mesh_file = f'../visualization/tmp/render_{object_name}.ply'
    ms.save_current_mesh(new_mesh_file)
    return new_mesh_file

def render(object_name, new_mesh_file, pic_i, floor, flip):
    dataset_dir = f'../datasets/mitsuba3_synthetic'
    scene_dict= {'type': 'scene'}
    scene_dict = load_integrator(scene_dict)
    scene_dict = load_object(scene_dict, new_mesh_file, flip)
    scene = mi.load_dict(scene_dict)
    
    bbox = scene.bbox()
    cameras_path = glob.glob(os.path.join(dataset_dir, object_name, '*', 'test', 'cameras_sphere.npz'))[0]
    cameras = np.load(cameras_path)

    world_mat = cameras['world_mat_{}'.format(pic_i)]
    P = world_mat
    P = P[:3, :4]
    _, pose = load_K_Rt_from_P(None,P)
    pose[:3, :1] = -pose[:3, :1]
    pose[:3, 1:2] = -pose[:3, 1:2]  # Flip the x+ and y+ to align coordinate system

    if floor:
        scene_dict = load_floor(scene_dict, bbox)
    sensor = load_sensor(T(pose), spp=256)
    scene_dict = load_emitter(scene_dict, pose)
    scene = mi.load_dict(scene_dict)

    noisy = mi.render(scene, sensor=sensor, spp=256)
    shape_n = noisy.shape[::-1]  
    denoiser = mi.OptixDenoiser(input_size=shape_n[1:3], albedo=False, normals=False, temporal=False)
    denoised = denoiser(noisy)
    image_np = mi.TensorXf(denoised).numpy()
    image = (image_np * 255 / np.max(image_np)).astype('uint8')
    
    return Image.fromarray(image).convert('RGB')


def main(input_dir, pic_i, floor, start, end):
    mi.set_variant("cuda_ad_rgb")
    os.makedirs(f'../visualization', exist_ok=True)
    column_gt = []
    column_neus2 = []
    column_instant_nsl_neus = []
    object_names = os.listdir(f'{input_dir}/neus2/CleanedMesh')
    print(object_names)
    print("start render...")
    for object_name in object_names[int(start):int(end)]:
        print(f"render scene: {object_name}")
        if pic_i == None:
            for i in tqdm(range(40)):
                # gt_mesh = os.path.join(f'{input_dir}/groundtruth', object_name, f'clean_{object_name}.ply')
                # column_gt.append(render(object_name, gt_mesh, i, floor))

                root_neus2_pr = os.path.join(input_dir, 'neus2', 'CleanedMesh', object_name)
                neus2_pr = glob.glob(os.path.join(root_neus2_pr, '*.ply'))[0]
                neus2_pr = recalculate_vertex_normals(neus2_pr, object_name)
                column_neus2.append(render(object_name, neus2_pr, i, floor))

                root_instant_pr = os.path.join(input_dir, 'NeRO', 'CleanedMesh', object_name)
                instant_pr = glob.glob(os.path.join(root_instant_pr, '*.ply'))[0]
                column_instant_nsl_neus.append(render(object_name, instant_pr, i, floor))
                
                imgs_2d = [column_gt, column_neus2, column_instant_nsl_neus]
                img_grid = ImageGrid()
                grid_image = img_grid.get_image_grid(imgs_2d)
                os.makedirs(f'../visualization/{object_name}', exist_ok=True)
                grid_image.save(f'../visualization/{object_name}/output_{object_name}_{i}.png')
                column_gt = []
                column_neus2 = []
                column_instant_nsl_neus = []
        else:
            gt = Image.open(os.path.join(f'../{object_name}.png'))
            column_gt.append(gt)

            root_neus2_pr = os.path.join(input_dir, 'neus2', 'CleanedMesh', object_name)
            neus2_pr = glob.glob(os.path.join(root_neus2_pr, '*.ply'))[0]
            neus2_pr = recalculate_vertex_normals(neus2_pr, object_name)
            column_neus2.append(render(object_name, neus2_pr, pic_i, floor, False))

            root_instant_pr = os.path.join(input_dir, 'NeRO', 'CleanedMesh', object_name)
            instant_pr = glob.glob(os.path.join(root_instant_pr, '*.ply'))[0]
            column_instant_nsl_neus.append(render(object_name, instant_pr, pic_i, floor, True))
            
    if pic_i:
        imgs_2d = [column_gt, column_neus2, column_instant_nsl_neus]
        img_grid = ImageGrid()
        grid_image = img_grid.get_image_grid(imgs_2d)
        grid_image.save('../visualization/output.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--floor', action='store_true', help='add floor or not, do not use it in dtu dataset')
    parser.add_argument('--pic_num', type=str, default=None, help='the number of pose you want to visualize.')
    parser.add_argument('--start', type=str, default='0')
    parser.add_argument('--end', type=str, default='3')
    parser.add_argument('--input_dir', type=str, help='location of your ply model')
    args = parser.parse_args()
    main(args.input_dir, args.pic_num, args.floor, args.start, args.end)