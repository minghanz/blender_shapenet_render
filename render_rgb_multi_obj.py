### 0. get the collection of objects and background images
### every iteration: 
### 1. sample a camera viewpoint
### 2. sample a list of objects
### 3. for each object, sample a pose
### 4. sample a background image
### 5. render

import sys
import os
import random
import bpy
import bmesh

from mathutils import Matrix
from mathutils import Vector

abs_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(abs_path))

from render_helper import *
from settings import *
import settings

from vp_generator import get_focal_len_from_obj_and_pose, get_bot_pts, project_pts_to_cam, get_4x4_RT_matrix_from_blender, get_K_from_blender, get_2d_bbox, write_cam_pose, write_obj_pose, write_pts

import time

from render_multi_obj_helper import TrafCamPoseSampler
import numpy as np 

from render_rgb import init_all, clear_mesh
from scipy.spatial.transform import Rotation

def set_cam_pose(R, t):
    ### set camera extrinsics
    ###The camera local coord is defined as x-right, y-up, z-back. right-down-front
    R_cam = R.transpose()
    t_cam = -R_cam.dot(t)

    R_cam[:,1] = -R_cam[:,1]
    R_cam[:,2] = -R_cam[:,2]

    r = Rotation.from_matrix(R_cam)
    euler_xyz = r.as_euler('xyz')   # blender takes radian as default: https://blender.stackexchange.com/questions/117325/how-do-i-set-my-camera-rotation-with-python

    cam_obj = bpy.data.objects['Camera']

    ### set
    cam_obj.location[0] = t_cam[0]
    cam_obj.location[1] = t_cam[1]
    cam_obj.location[2] = t_cam[2]
    cam_obj.rotation_euler[0] = euler_xyz[0]
    cam_obj.rotation_euler[1] = euler_xyz[1]
    cam_obj.rotation_euler[2] = euler_xyz[2]

    bpy.context.view_layer.update() # this is necessary for the change to be applied on cam_obj.matrix_world
    return

def set_cam_K_by_focal(f_factor):
    cam_obj = bpy.data.objects['Camera']

    sensor_height = cam_obj.data.sensor_height
    sensor_width = cam_obj.data.sensor_width
    cam_dim = sensor_height if sensor_height < sensor_width else sensor_width
    # new_focal_len = 300
    new_focal_len = cam_dim / 2.0 / f_factor

    ### set
    cam_obj.data.lens = new_focal_len

    K_homo = get_K_from_blender(cam_obj, bpy.context.scene, new_focal_len)
    K = K_homo[:,:3]

    bpy.context.view_layer.update()
    return K

def set_obj_pose(target_obj, yaw, position, scale):

    # pose_cur = np.array(target_obj.matrix_world)#.transpose()
    # print("pose_cur", pose_cur)
    pose_cur = np.array([[1,0,0,0], [0,0,-1,0], [0,1,0,0], [0,0,0,1]], dtype=np.float32)

    r = Rotation.from_euler('xyz', [0,0,yaw])
    R = r.as_matrix()
    T = np.hstack( (R, position.reshape(3,1)) ).astype(np.float32)
    T = np.vstack((T, np.array([[0,0,0,1]], dtype=np.float32)))
    T_scale = np.diag([scale,scale,scale,1]).astype(np.float32)

    pose_new = T @ T_scale @ pose_cur
    target_obj.matrix_world = pose_new.transpose()

    # target_obj.location[0] = pose_new[0,3]
    # target_obj.location[1] = pose_new[1,3]
    # target_obj.location[2] = pose_new[2,3]

    bpy.context.view_layer.update()
    

    return

if __name__ == "__main__":

    init_all()

    ### 0. get the collection of objects and background images
    obj_list = random_sample_objs(0)['car']
    # background_img_list = gen_list_of_valid_background_img()
    bg_folder = "/home/minghanz/Pictures/empty_road"
    background_img_list = [os.path.join(bg_folder, x) for x in os.listdir(bg_folder)]

    traf_cam_pose_sampler = TrafCamPoseSampler()


    ### 1. sample a camera viewpoint
    ### 2. sample a list of objects
    ### 3. for each object, sample a pose
    start_time = time.time()
    n_frames = 85
    bpy.context.scene.frame_set(9576)
    for i_frame in range(n_frames):
        clear_mesh()

        n_objs = random.randrange(1, 6)
        m_views = 5#random.randrange(3, 8)

        ### sample camera focal length
        focal_length_factor = random.uniform(0.1, 0.5)

        ### set the camera extrinsics
        # K, R, tvec, distCoeffs = traf_cam_pose_sampler.T_from_focal(focal_length_factor)
        K = set_cam_K_by_focal(focal_length_factor)
        R, tvec, distCoeffs = traf_cam_pose_sampler.T_from_K(K)
        set_cam_pose(R, tvec)

        for k_views in range(m_views):
            ### set up this frame
            current_frame = bpy.context.scene.frame_current
            file_output_node = bpy.context.scene.node_tree.nodes[4]
            
            ### gen txt file with the same name as the output image
            fpath = os.path.join(file_output_node.base_path, 'blender-{:06d}.color.txt'.format(current_frame))
            open(fpath, 'w').close()


            ### sample bg image
            bgimg_path = random.choice(background_img_list)
            with open(fpath, 'a') as f:
                f.write("bg_image_path: {}\n".format(bgimg_path))

            ### write cam pose
            cam_pos = get_4x4_RT_matrix_from_blender(bpy.data.objects['Camera'])
            K_homo = np.concatenate([K, np.zeros((3,1))], axis=1)
            write_cam_pose(cam_pos, K_homo, fpath)

            # n_objs = 3
            # ## sample in a batch
            # obj_list_cur = random.choices(obj_list, k=n_objs)
            # position_list_cur = [traf_cam_pose_sampler.samp_pts_3d() for i in range(n_objs)]
            # yaw_list_cur = [random.random()*2*np.pi for i in range(n_objs)]
            # for k_views in range(m_views):
            if k_views == 0:
                obj_path_dict = {}
                obj_name_list = [None]*n_objs
            for j_obj in range(n_objs):
                valid_obj = False
                while not valid_obj:
                    if k_views == 0:
                        ### sample object model, position and rotation
                        obj_cur = random.choice(obj_list)
                        ### import the object
                        bpy.ops.import_scene.obj(filepath=obj_cur)
                        target_obj = bpy.context.selected_objects[0]    ## selected_objects should have only one element after import 
                        obj_path_dict[target_obj.name] = obj_cur
                        obj_name_list[j_obj] = target_obj.name
                        fpath_valid = None
                    else:
                        obj_cur = obj_path_dict[obj_name_list[j_obj]]
                        target_obj = bpy.data.objects[obj_name_list[j_obj]]
                        valid_obj = True
                        fpath_valid = fpath

                    # for obj in bpy.data.objects:
                    #     print(obj.name)

                    ### sample position
                    position = traf_cam_pose_sampler.samp_pts_3d_from_2d(cam_pos, K_homo)
                    yaw = random.random()*2*np.pi
                    scale = random.uniform(3,8)

                    ### set position
                    set_obj_pose(target_obj, yaw, position, scale)
                    write_obj_pose(obj_cur, np.array(target_obj.matrix_world), fpath=fpath_valid, obj_id=j_obj)

                    ### get GT output
                    p_output = get_bot_pts(target_obj, fpath=fpath_valid, obj_id=j_obj)
                    pts_proj, pts_in_cam_ref = project_pts_to_cam(p_output, cam_pos, K_homo, fpath=fpath_valid, obj_id=j_obj)
                    
                    ### 2D bbox 
                    u_min, u_max, v_min, v_max = get_2d_bbox(target_obj, cam_pos, K_homo, fpath=fpath_valid, obj_id=j_obj)

                    if k_views == 0:
                        valid_obj = (u_max - u_min) < 2 * (pts_proj[0].max() - pts_proj[0].min())
                        if not valid_obj:
                            print("delete the invalid object!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            bpy.ops.object.delete()
                if k_views == 0:
                    write_obj_pose(obj_cur, np.array(target_obj.matrix_world), fpath=fpath, obj_id=j_obj)
                    write_pts(pts_proj, pts_in_cam_ref, u_min, u_max, v_min, v_max, fpath=fpath, obj_id=j_obj )

        
            if not os.path.exists(g_syn_rgb_folder):
                os.mkdir(g_syn_rgb_folder)

            ### load the background image
            image_node = bpy.context.scene.node_tree.nodes[0]
            image_node.image = bpy.data.images.load(bgimg_path)

            ### set the output file name
            file_output_node.file_slots[0].path = 'blender-######.color.png' # blender placeholder #

            ### start rendering
            bpy.ops.render.render(write_still=True)
            bpy.context.scene.frame_set(current_frame + 1)

        
    end_time = time.time()
    print("total_time", end_time-start_time)