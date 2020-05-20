import numpy as np 
import random

from render_helper import VP

import sys
sys.path.append("../")
from cvo_ops.utils_general.io import write_np_to_txt_like_kitti

import json
def read_bg_image_anno(fpath):
    with open(fpath) as f:
        load_dict = json.load(f)
    print(load_dict.keys)
    return load_dict

def gen_rand_vp():
    azimuth = random.random() * 360
    # azimuth = 0
    elevation = random.random() * 90
    # elevation = 0
    # tilt = random.uniform(-10, 10)
    tilt = 0
    distance = random.uniform(2, 10)
    vp = VP(azimuth, elevation, tilt, distance)

    return vp

def gen_vp_list(num_vp):
    vp_list = [gen_rand_vp() for i in range(num_vp)]
    return vp_list

def get_obj_dim(obj):
    pts = np.array([ obj.matrix_world @ v.co for v in obj.data.vertices])
    print("pts.shape", pts.shape)
    x_min = pts[:, 0].min()
    x_max = pts[:, 0].max()
    y_min = pts[:, 1].min()
    y_max = pts[:, 1].max()
    z_min = pts[:, 2].min()
    z_max = pts[:, 2].max()
    obj_dim = np.array([x_max-x_min, y_max-y_min, z_max-z_min])
    return obj_dim

def get_focal_len_from_obj_and_pose(obj, cam, view_dist=None, fpath=None):
    """This function is to get a proper focal length so that the object's size in image stays roughly consistent 
    while the size of objects and camera viewing distance changes drastically. """

    ### get the dimension of the object
    # dim = obj.dimensions    # width, height, length     # this is not true to the actualy mesh model sometimes
    dim = get_obj_dim(obj)     # this is not true to the actualy mesh model sometimes
    # dim[1] = 0.05
    # obj_max_dim = np.array(dim).max()
    obj_max_dim = np.sqrt((np.array(dim) ** 2).sum())
    # obj_max_dim = np.sqrt(dim[1]**2 + dim[2]**2)
    print(obj_max_dim)

    ### calculate the desired FOV with the object scale and camera distance
    if view_dist is not None:
        cam_dist = view_dist
        assert np.abs(cam_dist - np.sqrt(cam.location[0]**2 + cam.location[1]**2 + cam.location[2]**2)) < 0.01, \
                "{} {}".format(cam_dist, np.sqrt(cam.location[0]**2 + cam.location[1]**2 + cam.location[2]**2))
    else:
        cam_dist = np.sqrt(cam.location[0]**2 + cam.location[1]**2 + cam.location[2]**2)
    view_tan = obj_max_dim / 2 / cam_dist
    
    ### calculate focal length according to the target FOV
    sensor_height = cam.data.sensor_height
    sensor_width = cam.data.sensor_width
    cam_dim = sensor_height if sensor_height < sensor_width else sensor_width
    new_focal_len = cam_dim / 2 / view_tan
    cam.data.lens = new_focal_len

    ### save info to file
    if fpath is not None:
        with open(fpath, 'a') as f:
            f.write("cam_loc: {:.3f} {:.3f} {:.3f}\n".format(cam.location[0], cam.location[1], cam.location[2]))
            f.write("cam_dim: {:.3f} {:.3f} {:.3f}\n".format(sensor_height, sensor_width, cam.data.lens))
            f.write("veh_dim: {:.3f} {:.3f} {:.3f}\n".format(obj.dimensions[0], obj.dimensions[1], obj.dimensions[2]))  # object size from obj.dimensions
            f.write("veh_dim_mesh: {:.3f} {:.3f} {:.3f}\n".format(dim[0], dim[1], dim[2]))  # object size by going through the mesh vertices
            write_np_to_txt_like_kitti(f, np.array(obj.matrix_world), "veh_pos")            # transformation from object local frame to global frame x_glob = H * x_local

    return new_focal_len

def get_bot_pts(obj, fpath=None):
    """This function obtains the lowest points on each tire as feature point of the vehicle object"""

    ### get the mesh vertices of the object
    vertices = obj.data.vertices
    pts = np.array([ v.co for v in vertices])   # x-right, y-up, z-back (see README of ShapeNetCore), shape: N*3 
    # pts = np.array([(obj.matrix_world @ v.co) for v in vertices])   # x-right, y-front, z-up, shape: N*3 
    print("pts.shape", pts.shape)
    print("pts.dtype", pts.dtype)
    ## obj.matrix_world (4*4) is a Matrix type, vertices[0].co (3) is a Vector type, both of blender specific type, not numpy type

    ### get the lowest point of each quater of the vehicle
    front_mask = pts[:,2]<0
    left_mask = pts[:,0]<0
    pts_front_left = pts[front_mask & left_mask, :]
    pts_back_left = pts[~front_mask & left_mask, :]
    pts_front_right = pts[front_mask & ~left_mask, :]
    pts_back_right = pts[~front_mask & ~left_mask, :]

    pt_front_left = pts_front_left[pts_front_left[:,1]==pts_front_left[:,1].min(), :]
    pt_back_left = pts_back_left[pts_back_left[:,1]==pts_back_left[:,1].min(), :]
    pt_front_right = pts_front_right[pts_front_right[:,1]==pts_front_right[:,1].min(), :]
    pt_back_right = pts_back_right[pts_back_right[:,1]==pts_back_right[:,1].min(), :]
    ## usually more than one point is selected

    p_front_left = pt_front_left.mean(axis=0)
    p_back_left = pt_back_left.mean(axis=0)
    p_front_right = pt_front_right.mean(axis=0)
    p_back_right = pt_back_right.mean(axis=0)

    ### transform to global coordinate
    # p_output = np.concatenate([pt_front_left, pt_back_left, pt_front_right, pt_back_right], axis=0).T
    p_output = np.stack([p_front_left, p_back_left, p_front_right, p_back_right], axis=1) # 3*N
    p_output_homo = np.concatenate([ p_output, np.ones((1,p_output.shape[1])) ], axis=0) # 4*N
    p_output_homo = np.array(obj.matrix_world) @ p_output_homo
    
    return p_output_homo

def project_pts_to_cam(p_output_homo, cam_pos, K_homo, fpath=None):
    """Project point coordinates in 3D world space to image coordinates"""

    ### transform to camera coordinate (3D)
    pts_in_cam_ref = cam_pos @ p_output_homo
    ### project to image coordinate
    pts_proj = np.matmul(K_homo, pts_in_cam_ref)
    pts_proj = pts_proj / pts_proj[[2], :]

    ### write to file
    if fpath is not None:
        with open(fpath, 'a') as f:
            write_np_to_txt_like_kitti(f, cam_pos, "cam_pos_inv")
            write_np_to_txt_like_kitti(f, pts_in_cam_ref.T, "pts_in_cam")
            write_np_to_txt_like_kitti(f, K_homo, "K")
            write_np_to_txt_like_kitti(f, pts_proj.T, "pts_proj")
            # f.write("cam_pos:\n")
            # f.write(np.array_str(np.array(cam_obj.matrix_world)) )

    return pts_proj

def get_2d_bbox(obj, cam_pos, K_homo, obj_pose=None, fpath=None):
    if obj_pose is None:
        pts = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])    # N*3
        pts_homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1).T # 4*N
    else:
        pts = np.array([ v.co for v in obj.data.vertices]) #N*3
        pts_homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1).T # 4*N
        pts_homo = np.matmul(obj_pose, pts_homo)

    pts_proj = project_pts_to_cam(pts_homo, cam_pos, K_homo)

    u_min = pts_proj[0].min()
    u_max = pts_proj[0].max()
    v_min = pts_proj[1].min()
    v_max = pts_proj[1].max()

    if fpath is not None:
        with open(fpath, 'a') as f:
            f.write("bbox_uuvv: {} {} {} {}\n".format(u_min, u_max, v_min, v_max))

    return u_min, u_max, v_min, v_max


def get_4x4_RT_matrix_from_blender(cam):
    """The camera local coord is defined as x-right, y-up, z-back. 
    Therefore besides the cam.matrix_world, you still need to flip y and z axis to make it consistent to 
    what is generally perceived in a regular camera intrinsic matrix. """

    ### These three produces identical results
    ### opt 1: use blender Matrix and Vector type
    # from render_rgb import get_3x4_RT_matrix_from_blender_nnp
    # RT = get_3x4_RT_matrix_from_blender_nnp(cam)
    # RT = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    # return np.array(RT)

    ### opt 2: mix blender Matrix/Vector and np array
    # RT = get_4x4_RT_matrix_from_blender_np(cam)

    ### opt 3: use np array
    RT = get_4x4_RT_matrix_from_blender_np_simple(cam)

    return RT

def get_4x4_RT_matrix_from_blender_np_simple(cam):
    RT = np.array(cam.matrix_world) # 4*4
    RT = np.linalg.inv(RT)
    RT[1] = -RT[1]
    RT[2] = -RT[2]
    return RT

def get_4x4_RT_matrix_from_blender_np(cam):
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_world2bcam
    T_world2cv = T_world2bcam
    R_world2cv[1] = -R_world2cv[1]
    R_world2cv[2] = -R_world2cv[2]
    T_world2cv[1] = -T_world2cv[1]
    T_world2cv[2] = -T_world2cv[2]

    # # put into 4x4 matrix
    R = np.array(R_world2cv)
    T = np.array(T_world2cv).reshape(-1, 1)
    RT = np.concatenate([R, T], axis=1)
    RT = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    print('RT.shape', RT.shape)
    return RT

def get_K_from_blender(cam_obj, scene, new_len):
    """Get camera intrinsic matrix from blender camera object. """
    ### get the intrinsic matrix of the camera. Only get_calibration_matrix_K_from_blender_2 works. 
    # K = get_calibration_matrix_K_from_blender(cam_obj.data, scene, new_len)
    # K = get_calibration_matrix_K_from_blender_my(cam_obj.data, scene, new_len)
    K = get_calibration_matrix_K_from_blender_2(cam_obj.data, scene, new_len)

    K_homo = np.concatenate([K, np.zeros((3,1))], axis=1)
    return K_homo

### This function does not provide correct intrinsic matrix
def get_calibration_matrix_K_from_blender_my(camd, scene, new_len):
    f_in_mm = camd.lens
    print("{}, {}".format(f_in_mm, new_len) )

    assert abs(f_in_mm - new_len) < 0.01, "{}, {}".format(f_in_mm, new_len)

    im_w_px = scene.render.resolution_x
    im_h_px = scene.render.resolution_y
    im_w_mm = camd.sensor_width
    im_h_mm = camd.sensor_height
    im_cw_px = im_w_px / 2
    im_ch_px = im_h_px / 2
    im_cw_mm = im_w_mm / 2
    im_ch_mm = im_h_mm / 2

    w_px_by_mm = im_w_px / im_w_mm
    h_px_by_mm = im_h_px / im_h_mm
    
    im_fw_px = w_px_by_mm * f_in_mm
    im_fh_px = h_px_by_mm * f_in_mm

    K = np.array(
        [[im_fw_px, 0,    im_cw_px],
        [    0  ,  im_fh_px, im_ch_px],
        [    0  ,    0,      1 ]])

    return K

### This function does not provide correct intrinsic matrix
def get_calibration_matrix_K_from_blender(camd, scene, new_len):
    '''https://blender.stackexchange.com/a/38189'''
    f_in_mm = camd.lens
    print("{}, {}".format(f_in_mm, new_len) )

    assert abs(f_in_mm - new_len) < 0.1, "{}, {}".format(f_in_mm, new_len)
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    # K = Matrix(
    #     ((alpha_u, skew,    u_0),
    #     (    0  ,  alpha_v, v_0),
    #     (    0  ,    0,      1 )))

    K = np.array(
        [[alpha_u, skew,    u_0],
        [    0  ,  alpha_v, v_0],
        [    0  ,    0,      1 ]])
    return K



############# below from https://blender.stackexchange.com/a/120063
# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender_2(camd, scene, new_len):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    f_in_mm = camd.lens
    assert abs(f_in_mm - new_len) < 0.01, "{}, {}".format(f_in_mm, new_len)

    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    # K = Matrix(
    #     ((s_u, skew, u_0),
    #     (   0,  s_v, v_0),
    #     (   0,    0,   1)))
    
    K = np.array(
        [[s_u, skew,    u_0],
        [    0  ,  s_u, v_0],
        [    0  ,    0,      1 ]])
    return K