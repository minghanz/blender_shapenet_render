### distortion effect is ignored, how large is the error? 
### tutorial: https://docs.opencv.org/master/d9/dab/tutorial_homography.html

import numpy as np 
import cv2
import random

from vp_generator import project_pts_to_cam

import sys
sys.path.append("../")
from bev.bev import BEVWorldSpec
from bev.calib import Calib
from bev.homo import Rt_from_pts_K_dist, Rt_from_homo_K
from bev.constructor.homo_constr import preset_bspec, preset_calib

def load_pts_world_to_bevimg(u_size, v_size, name, sub_id=None):
    assert name in ["lturn", "KoPER"]
    if name == "lturn":
        ### copied from trafcam_proc
        x_min = -10
        x_max = 42 #30
        y_min = -37#-30
        y_max = 31

        # u_min = 0
        # u_max = u_size
        # v_min = 0
        # v_max = v_size
        # pts_world = np.array([[x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]], dtype=np.float)
        # pts_img = np.array([[u_min, v_min], [u_min, v_max], [u_max, v_max], [u_max, v_min]], dtype=np.float)
        # H, mask = cv2.findHomography(pts_world, pts_img)
        # print(H)

        bspec = BEVWorldSpec(u_size=u_size, v_size=v_size, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, u_axis="-x", v_axis="y")
    else:
        vuratio = float(v_size)/ u_size
        if sub_id == 1:
            x_size = 60
            x_max = 45
            y_min = -30
            u_axis = "-x"
            v_axis = "y"
        elif sub_id == 4:
            x_size = 50
            x_max = 30
            y_min = -13
            u_axis = "x"
            v_axis = "-y"
        else:
            raise ValueError("sub_id not recogized")
        y_size = vuratio * x_size
        bspec = BEVWorldSpec(u_size=u_size, v_size=v_size, x_max=x_max, x_size=x_size, y_min=y_min, y_size=y_size, u_axis=u_axis, v_axis=v_axis)

    H_world_bev = bspec.gen_H_world_bev()
    H_bev_world = np.linalg.inv(H_world_bev)

    return H_bev_world

def load_T(name, sub_id):
    if name == "KoPER":
        assert sub_id in [1,4]
        if sub_id == 1:
            T = np.array([-0.998701024990115, -0.0243198486052637, 0.0447750784199287, 12.0101713926222,
                            -0.0488171908715811, 0.708471062121443, -0.704049455657713, -3.19544493781655,
                            -0.0145994711925239, -0.705320706558607, -0.708738002607851, 18.5953697835002,
                            0, 0, 0, 1], dtype=np.float).reshape(4,4)
            calib = Calib(fx=336.2903, fy=335.5113, cx=321.3685, cy=251.1326, T=T)
        else:
            T = np.array([0.916927873702706, 0.399046485499693, -0.00227526644131446, -9.32023173352383,
                            0.287745260046085, -0.665109208127767, -0.689080841835460, -5.17417993923343,
                            -0.276488588820672, 0.631182733979628, -0.724680906729269, 17.1155540514235,
                            0, 0, 0, 1], dtype=np.float).reshape(4,4)
            
            calib = Calib(fx=331.2292, fy=330.4413, cx=325.4500, cy=252.1456, T=T)
        H_world_img = calib.gen_H_world_img(mode="from_KRt")
    else:
        raise ValueError("cam_name not recognized", name)

    return H_world_img

def load_pts(name, img_width, img_height):
    if name == "lturn":
        ### load homographic calibration and draw on picture
        pts_3d = np.array( [[0, 0, 0], [3.7, 0, 0], [7.4, 0, 0], [-0.87, 21.73, 0],
                        [-4.26, 21.73, 0], [-5.22, -2.17, 0]], dtype=np.float32)
        pts_world = pts_3d[:,:2]

        pts_2d = np.array([[1575, 611], [1428, 608], [1256, 605], [1066, 876],
                        [1368, 924], [1866, 601]], dtype=np.float32)
        ## looks like these points are captured on 1080p screen capture
        pts_2d[:,0] = pts_2d[:,0] / 1920 * img_width
        pts_2d[:,1] = pts_2d[:,1] / 1080 * img_height
        pts_cam = pts_2d

    else:
        raise ValueError("cam_name not recognized", name)
    return pts_3d, pts_2d

def get_position_corner(pts_2d, pts_3d, width, height):
    H, mask = cv2.findHomography(pts_2d, pts_3d[:,:2])

    # vanish_u = None
    # vanish_v = None


    corners_2d = np.array([[0,height*0.5], [0,height], [width, height], [width, height*0.5]], dtype=np.float32)
    corners_3d = cv2.perspectiveTransform(corners_2d[np.newaxis], H)[0] # perspectiveTransform requires 3-dim input: https://stackoverflow.com/a/45818765

    return H, corners_3d

def load_K(f_coef, img_width, img_height):
    ### f_coef ~ [1,1.5]
    K = np.eye(3)
    K[0,2] = img_width/2
    K[1,2] = img_height/2
    f = img_width/f_coef
    K[0,0] = f
    K[1,1] = f
    distCoeffs = np.zeros((4), dtype=np.float32)

    return K, distCoeffs


class TrafCamPoseSampler:
    def __init__(self, name, sub_id=None):
        assert name in ["lturn", "KoPER"]
        self.name = name

        ######################### use preset
        ### world-img
        self.calib = preset_calib(dataset_name=name, sub_id=sub_id)
        self.H_world_img = self.calib.gen_H_world_img(self.calib.mode)
        self.H_img_world = np.linalg.inv(self.H_world_img)
        ### world-bev
        bspec = preset_bspec(dataset_name=name, sub_id=sub_id)
        self.H_world_bev = bspec.gen_H_world_bev()
        self.H_bev_world = np.linalg.inv(self.H_world_bev)

        self.width = self.calib.u_size
        self.height = self.calib.v_size
        self.bev_width = bspec.u_size
        self.bev_height = bspec.v_size
        if self.name == "lturn":
            self.pts_3d, self.pts_2d = self.calib.pts_world, self.calib.pts_image
            self.bg_path = "/home/minghanz/Pictures/empty_road"
            # self.bg_path = "/home/minghanz/Pictures/env_texture"
            self.bev_mask_path = "/home/minghanz/Pictures/empty_road/bev/road_mask_bev.png"
            self.ori_mask_path = "/home/minghanz/Pictures/empty_road/mask/road_mask.png"
            self.fix_yaw = None
        elif self.name == "KoPER":
            # self.bg_path = "/media/sda1/datasets/extracted/KoPER/added/background/SK_{}".format(sub_id)
            self.bg_path = "/home/minghanz/Pictures/env_texture"
            self.bev_mask_path = "/media/sda1/datasets/extracted/KoPER/added/masks/SK_{}_bev.png".format(sub_id)
            self.ori_mask_path = "/media/sda1/datasets/extracted/KoPER/added/masks/SK_{}.png".format(sub_id)
            self.fix_yaw = None #0

            # self.bev_mask_path = "/media/sda1/datasets/extracted/KoPER/added/masks/SK_{}_direction1_bev.png".format(sub_id)
            # self.ori_mask_path = "/media/sda1/datasets/extracted/KoPER/added/masks/SK_{}_direction1.png".format(sub_id)
            # self.fix_yaw = 2.454 -np.pi/2   ### the yaw is in world coordinate, which is shared across sub_ids

            # self.bev_mask_path = "/media/sda1/datasets/extracted/KoPER/added/masks/SK_{}_direction2_bev.png".format(sub_id)
            # self.ori_mask_path = "/media/sda1/datasets/extracted/KoPER/added/masks/SK_{}_direction2.png".format(sub_id)
            # self.fix_yaw = -2.281 -np.pi/2
        else:
            raise ValueError("name not recognized", name)

        self.bev_mask = cv2.imread(self.bev_mask_path)

        assert self.bev_height == self.bev_mask.shape[0]
        assert self.bev_width == self.bev_mask.shape[1]
        
        ####################################

        # ############ not using preset
        # if self.name == "lturn":
        #     self.width = 852
        #     self.height = 480

        #     #### image to world homography
        #     self.pts_3d, self.pts_2d = load_pts(self.name, self.width, self.height)

        #     # self.H_world_img, self.corners_3d = get_position_corner(self.pts_2d, self.pts_3d, self.width, self.height)
        #     self.calib = Calib(pts_world=self.pts_3d, pts_image=self.pts_2d)
        #     self.H_world_img = self.calib.gen_H_world_img(mode="from_pts")
        #     self.H_img_world = np.linalg.inv(self.H_world_img)

        #     #### world to bev homography
        #     self.bev_mask = cv2.imread("/home/minghanz/Pictures/empty_road/bev/road_mask_bev.png")
        #     self.bev_width = self.bev_mask.shape[1] #416
        #     self.bev_height = self.bev_mask.shape[0] #544#480 #544 # 624

        #     self.H_bev_world = load_pts_world_to_bevimg(self.bev_width, self.bev_height, self.name)
        #     self.H_world_bev = np.linalg.inv(self.H_bev_world)

        # elif self.name == "KoPER":
        #     self.width = 656
        #     self.height = 494
        #     sub_id = 4
        #     assert sub_id in [1,4]

        #     #### image to world homography
        #     self.H_world_img = load_T(self.name, sub_id=sub_id)
        #     self.H_img_world = np.linalg.inv(self.H_world_img)

        #     #### world to bev homography
        #     self.bev_mask = cv2.imread("/media/sda1/datasets/extracted/KoPER/added/SK_{}_empty_road_mask_bev.png".format(sub_id))
        #     self.bev_width = self.bev_mask.shape[1] #416
        #     self.bev_height = self.bev_mask.shape[0] #544#480 #544 # 624

        #     self.H_bev_world = load_pts_world_to_bevimg(self.bev_width, self.bev_height, self.name, sub_id)
        #     self.H_world_bev = np.linalg.inv(self.H_bev_world)

        # else:
        #     raise ValueError("name not recognized")

    def get_bg_path(self):
        return self.bg_path

    def get_mask_path(self):
        return self.ori_mask_path
    
    def T_from_focal(self, f_coef):

        ### given focal length, calc RT matrix
        K, distCoeffs = load_K(f_coef, self.width, self.height)

        if hasattr(self, "pts_3d") and hasattr(self, "pts_2d"):
            R, tvec = Rt_from_pts_K_dist(pts_world=self.pts_3d, pts_img=self.pts_2d, K=K, dist_coeffs=distCoeffs)
        else:
            R, tvec = Rt_from_homo_K(H_img_world=self.H_img_world, K=K)

        # retval, rvec, tvec = cv2.solvePnP(self.pts_3d, self.pts_2d, K, distCoeffs)
        # R, jacobian	= cv2.Rodrigues(rvec)
        ### https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d

        # pts_proj = K.dot(R.dot(pts_3d.transpose())+tvec).transpose()
        # pts_proj = pts_proj / pts_proj[:,[2]]

        return K, R, tvec, distCoeffs

    def T_from_K(self, K):
        ### the K from load_K function may not be what blender is actually using, therefore directly taking what blender is using as input

        distCoeffs = np.zeros((4), dtype=np.float32)

        if hasattr(self, "pts_3d") and hasattr(self, "pts_2d"):
            R, tvec = Rt_from_pts_K_dist(pts_world=self.pts_3d, pts_img=self.pts_2d, K=K, dist_coeffs=distCoeffs)
        else:
            # print("self.H_img_world", self.H_img_world)
            # print("K", K)
            R, tvec = Rt_from_homo_K(H_img_world=self.H_img_world, K=K)

        # retval, rvec, tvec = cv2.solvePnP(self.pts_3d, self.pts_2d, K, distCoeffs)
        # R, jacobian	= cv2.Rodrigues(rvec)

        return R, tvec, distCoeffs

    def samp_yaw(self):
        yaw = random.random()*2*np.pi if self.fix_yaw is None else self.fix_yaw + random.randint(0,1)*np.pi
        return yaw

    def samp_pts_3d_from_bev(self):
        ### advantage of this sampling strategy is that we can use the bev mask to finely control the valid region, 
        ### and encourage more sampling at remote region compared with sampling from original image (samp_pts_3d_from_2d).
        ### this sampling strategy makes sure the vehicle is in bev, but may be slightly out of original image 
        valid = False
        while not valid:
            x = int(random.random() * self.bev_width)
            y = int(random.random() * self.bev_height)
            if self.bev_mask[y,x,0]>0:
                # valid = True
                pt_world = cv2.perspectiveTransform(np.array([[[x,y]]], dtype=np.float32), self.H_world_bev)[0,0]
                pt_img = cv2.perspectiveTransform(np.array([[[x,y]]], dtype=np.float32), self.H_img_world.dot(self.H_world_bev))[0,0]   
                # pt_img is according to the fixed homography, may be slightly different from the actual projection calculated using the K, Rt
                # we calculate this only to make sure the calculation is correct, otherwise the assert is redundent
                u_img = pt_img[0]
                v_img = pt_img[1]
                # assert u_img >=0 and u_img <self.width and v_img>=0 and v_img <self.height, "{}, {}, {}, {}".format(u_img, v_img, self.width, self.height)
                x_world = pt_world[0]
                y_world = pt_world[1]
                valid = u_img >=0 and u_img <self.width and v_img>=0 and v_img <self.height
                if not valid:
                    print("not valid: {}, {}, {}, {}, {}, {}, {}, {}".format(x,y,u_img, v_img, self.width, self.height, x_world, y_world))
        
        return np.array([x_world, y_world, 0], dtype=np.float32)

    def samp_pts_3d(self):
        min_x = self.corners_3d[:,0].min()
        max_x = self.corners_3d[:,0].max()
        min_y = self.corners_3d[:,1].min()
        max_y = self.corners_3d[:,1].max()
        
        in_pic = False
        while not in_pic:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            p_uv = cv2.perspectiveTransform(np.array([[[x,y]]], dtype=np.float32), self.H_img_world)[0,0]
            in_pic = p_uv[0] > 0 and p_uv[0] < self.width-1 and p_uv[1] > 0 and p_uv[1] < self.height-1

        return np.array([x, y, 0], dtype=np.float32)


    def samp_pts_3d_from_2d(self, cam_pos, K_homo):
        valid = False
        while not valid:
            u = random.uniform(0, self.width)
            v = random.uniform(self.height*0.5, self.height)

            p_xy = cv2.perspectiveTransform(np.array([[[u,v]]], dtype=np.float32), self.H_world_img)[0,0]

            p_xy_3d = np.array([[p_xy[0], p_xy[1], 0, 1]], dtype=np.float32).reshape(4,1)
            pts_proj, pts_in_cam_ref = project_pts_to_cam(p_xy_3d, cam_pos, K_homo)
            valid = pts_proj[0] > 0 and pts_proj[0] < self.width and pts_proj[1] > 0 and pts_proj[1] < self.height

        return p_xy_3d[:3].reshape(-1)

    # def check_in_view(self, u_min, u_max, v_min, v_max):
    #     valid = u_min < self.width and v_min < self.height and u_max > 0 and v_max > 0
    #     return valid