import sys
import os
import random
import pickle
import bpy
import bmesh

from mathutils import Matrix
from mathutils import Vector

abs_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(abs_path))

from render_helper import *
from settings import *
import settings

import time

from render_rgb import init_all, clear_mesh
from vp_generator import get_2d_bbox

sys.path.append("..")
from cvo_ops.utils_general.io import read_calib_file

import numpy as np
# import cv2

def get_target_obj(): 
    for ob in bpy.context.scene.objects:
    # whatever objects you want to join...
        if ob.type == 'MESH':
            obj_target = ob
    return obj_target

def get_bbox_edge(obj, obj_pose, cam_pose_inv, cam_K):
    pts = np.array([ v.co for v in obj.data.vertices]) #N*3
    pts_homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)

    pts_world = np.matmul(obj_pose, pts_homo.T)
    pts_ref_cam = cam_pose_inv.dot(pts_world)

    pts_proj = np.matmul(cam_K, pts_ref_cam)
    pts_proj = pts_proj / pts_proj[[2]]

    u_min = pts_proj[0].min()
    u_max = pts_proj[0].max()
    v_min = pts_proj[1].min()
    v_max = pts_proj[1].max()

    return u_min, u_max, v_min, v_max

def load_info(fpath):
    finfo = read_calib_file(fpath)
    cam_pose_inv = finfo["cam_pos_inv"].reshape((4,4))
    K = finfo["K"].reshape((3,4))
    model_path = finfo["model_path"]
    model_pose = finfo["veh_pos"].reshape((4,4))

    return model_path, model_pose, cam_pose_inv, K, finfo
    
if __name__ == "__main__":
    #### init scenario
    start_time = time.time()
    init_all()

    #### load model and camera info from file
    data_root = "/media/sda1/datasets/extracted/shapenet_render/syn_rgb/car"
    txt_files = [f for f in os.listdir(data_root) if f.endswith(".txt")]
    txt_files = sorted(txt_files)

    cur_model_path = None
    cur_time = time.time()
    for tf in txt_files:
        print(tf)
        tf_path = os.path.join(data_root, tf)
        model_path, model_pose, cam_pose_inv, K, finfo = load_info(tf_path)
        if "bbox" in finfo:
            print("already processed, skipping:", tf)
            continue

        if model_path != cur_model_path:
            ## check the pose is the same as
            ## import the obj
            clear_mesh()
            bpy.ops.import_scene.obj(filepath=model_path)
            cur_model_path = model_path
            obj_target = get_target_obj()

            obj_pose = np.array(obj_target.matrix_world)
            assert np.abs(obj_pose - model_pose).max() < 1e-3

        ### get_bbox_edge can be discarded since get_2d_bbox did the same.
        # u_min, u_max, v_min, v_max = get_bbox_edge(obj_target, model_pose, cam_pose_inv, K)
        u_min, u_max, v_min, v_max = get_2d_bbox(obj_target, cam_pose_inv, K, model_pose)
        print(u_min, u_max, v_min, v_max)

        with open(tf_path, 'a') as f:
            f.write("bbox: {} {} {} {}\n".format(u_min, u_max, v_min, v_max))
        
        new_cur_time = time.time()
        print("one loop: {:.2f}s".format(new_cur_time - cur_time))
        cur_time = new_cur_time
        
    print("Total time: {:.2f}s".format(time.time() - start_time))