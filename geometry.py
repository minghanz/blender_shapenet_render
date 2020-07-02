"""This file includes functions to check occlusion and intersection, so as to create more real synthetic images"""

import numpy as np

def check_project_intersect(pmin_1, pmax_1, pmin_2, pmax_2):
    small_1 = pmin_1 < pmin_2
    if small_1:
        overlap = pmax_1 > pmin_2
    else:
        overlap = pmax_2 > pmin_1
    return overlap

def project_to_nvector(nvec, poly):
    pmin = (poly[0] * nvec).sum()
    pmax = pmin
    for pt in poly:
        proj = (pt * nvec).sum()
        if proj < pmin:
            pmin = proj
        elif proj > pmax:
            pmax = proj

    return pmin, pmax

def check_poly_intersect(poly_1, poly_2):
    """poly is a n(4)*2 nparray"""
    ### reference: https://www.codeproject.com/Articles/15573/2D-Polygon-Collision-Detection
    overlap_flag = True
    #### loop over each edge
    for poly_cur in [poly_1, poly_2]:
        npt = poly_cur.shape[0]
        for pid, pt in enumerate(poly_cur):
            #### calculate the projection axis
            pt_next = poly_cur[ (pid+1) % npt ]
            vector = pt_next - pt
            vector_perp_norm = np.array([ -vector[1], vector[0] ]) / (np.sqrt(vector[0]**2+vector[1]**2 ) + 1e-6 )

            #### project both polygons to the axis
            pmin_1, pmax_1 = project_to_nvector(vector_perp_norm, poly_1)
            pmin_2, pmax_2 = project_to_nvector(vector_perp_norm, poly_2)

            #### check overlap in the projection
            overlap_flag = check_project_intersect(pmin_1, pmax_1, pmin_2, pmax_2)
            if not overlap_flag:
                break
        if not overlap_flag:
            break
    
    return overlap_flag

def check_occlusion(cam_bbox_target, cam_bbox_env, ctr_target, ctr_env, ctr_cam):
    """bbox is 4*2 nparray, ctr is 3D coordinate
    return True if target is totally inside env and it is behind env"""
    ### reference: https://demonstrations.wolfram.com/AnEfficientTestForAPointToBeInAConvexPolygon/
    ### x_i+1 * y_i - x_i * y_i+1
    n_target = cam_bbox_target.shape[0]
    n_env = cam_bbox_env.shape[0]
    flag_signs = []
    for i in range(n_target):
        x0 = cam_bbox_target[i,0]
        y0 = cam_bbox_target[i,1]

        for j in range(n_env):
            x_cur = cam_bbox_env[j,0] - x0
            y_cur = cam_bbox_env[j,1] - y0

            x_next = cam_bbox_env[(j+1)%n_env, 0] - x0
            y_next = cam_bbox_env[(j+1)%n_env, 1] - y0
            
            flag_signs.append(x_next * y_cur - x_cur * y_next)
    
    flag_signs = np.array(flag_signs)
    inside = (flag_signs>0).all() or (flag_signs<0).all()

    if not inside:
        return False
    
    dist_to_cam_target = ((ctr_target - ctr_cam)**2).sum()
    dist_to_cam_env = ((ctr_env - ctr_cam)**2).sum()

    if dist_to_cam_env > dist_to_cam_target:
        return False

    return True


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from cvo_ops.utils_general.io import read_calib_file

    # poly_1 = np.array([[1,1], [1,2], [2,2], [2,1]])
    # # poly_2 = np.array([[1.5, 1.5], [0.5, 2.5], [1, 3], [2,2]])
    # poly_2 = np.array([[15, 15], [5, 25], [10, 30], [22,22]])

    # overlap_flat = check_poly_intersect(poly_1, poly_2)
    # print(overlap_flat)


    img_path = '/media/sda1/datasets/extracted/shapenet_lturn_3D_test/syn_rgb/blender-{:06d}.color.png'
    txt_path = '/media/sda1/datasets/extracted/shapenet_lturn_3D_test/syn_rgb/blender-{:06d}.color.txt'

    for fid in range(56, 61):
        txt_cur = txt_path.format(fid)
        img_cur = img_path.format(fid)

        info = read_calib_file(txt_cur)

        bbox3ds = [info[key].reshape(8, 4) for key in info if "3dbox" in key]
        ground_bboxs = [bbox[:4, :2] for bbox in bbox3ds] # a list of ground bbox of 4*2

        bbox_cams = [info[key] for key in info if "bbox" in key] # a list of cam 2dbbox of 4 (uuvv)
        bbox_cams = [np.array([[bbox[0], bbox[2]], [bbox[0], bbox[3]], [bbox[1], bbox[3]], [bbox[1], bbox[2]] ]) for bbox in bbox_cams ]

        poses = [info[key].reshape(4,4) for key in info if "obj_pose" in key]
        centers = [pose[:3, 3] for pose in poses] # a list of points of 3 (xyz)

        cam_pose_inv = info["cam_pos_inv"].reshape(4,4)
        cam_pose = np.linalg.inv(cam_pose_inv)
        cam_center = cam_pose[:3, 3] # cam position of 3 (xyz)
        # cam_focal = info

        assert len(ground_bboxs) == len(centers) and len(centers) == len(bbox_cams)
        n_obj = len(ground_bboxs)

        print("frame {}".format(fid))
        for obj_id_1 in range(n_obj):
            for obj_id_2 in range(obj_id_1+1, n_obj):
                intersect = check_poly_intersect(ground_bboxs[obj_id_1], ground_bboxs[obj_id_2])
                occluded = check_occlusion(bbox_cams[obj_id_1], bbox_cams[obj_id_2], centers[obj_id_1], centers[obj_id_2], cam_center)
                print("{} by {}: intersection: {}, occlusion: {}".format(obj_id_1, obj_id_2, intersect, occluded))


        for obj_id_2 in range(n_obj):
            for obj_id_1 in range(obj_id_2+1, n_obj):
                occluded = check_occlusion(bbox_cams[obj_id_1], bbox_cams[obj_id_2], centers[obj_id_1], centers[obj_id_2], cam_center)
                print("{} by {}: intersection: {}, occlusion: {}".format(obj_id_1, obj_id_2, intersect, occluded))