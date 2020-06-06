### distortion effect is ignored, how large is the error? 
### tutorial: https://docs.opencv.org/master/d9/dab/tutorial_homography.html

import numpy as np 
import cv2
import random

from vp_generator import project_pts_to_cam

def load_pts(name, img_width, img_height):
    if name == "lturn":
        ### load homographhic calibration and draw on picture
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
        raise ValueError("cam_name not recognized")
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
    def __init__(self):
        self.name = "lturn"
        self.width = 852
        self.height = 480

        self.pts_3d, self.pts_2d = load_pts(self.name, self.width, self.height)
        self.H, self.corners_3d = get_position_corner(self.pts_2d, self.pts_3d, self.width, self.height)
        self.H_inv = np.linalg.inv(self.H)
    
    def T_from_focal(self, f_coef):

        ### given focal length, calc RT matrix
        K, distCoeffs = load_K(f_coef, self.width, self.height)

        retval, rvec, tvec = cv2.solvePnP(self.pts_3d, self.pts_2d, K, distCoeffs)

        R, jacobian	= cv2.Rodrigues(rvec)

        # pts_proj = K.dot(R.dot(pts_3d.transpose())+tvec).transpose()
        # pts_proj = pts_proj / pts_proj[:,[2]]

        return K, R, tvec, distCoeffs

    def T_from_K(self, K):
        ### the K from load_K function may not be what blender is actually using, therefore directly taking what blender is using as input

        distCoeffs = np.zeros((4), dtype=np.float32)

        retval, rvec, tvec = cv2.solvePnP(self.pts_3d, self.pts_2d, K, distCoeffs)

        R, jacobian	= cv2.Rodrigues(rvec)

        return R, tvec, distCoeffs

    def samp_pts_3d(self):
        min_x = self.corners_3d[:,0].min()
        max_x = self.corners_3d[:,0].max()
        min_y = self.corners_3d[:,1].min()
        max_y = self.corners_3d[:,1].max()
        
        in_pic = False
        while not in_pic:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            p_uv = cv2.perspectiveTransform(np.array([[[x,y]]], dtype=np.float32), self.H_inv)[0,0]
            in_pic = p_uv[0] > 0 and p_uv[0] < self.width-1 and p_uv[1] > 0 and p_uv[1] < self.height-1

        return np.array([x, y, 0], dtype=np.float32)


    def samp_pts_3d_from_2d(self, cam_pos, K_homo):
        valid = False
        while not valid:
            u = random.uniform(0, self.width)
            v = random.uniform(self.height*0.5, self.height)

            p_xy = cv2.perspectiveTransform(np.array([[[u,v]]], dtype=np.float32), self.H)[0,0]

            p_xy_3d = np.array([[p_xy[0], p_xy[1], 0, 1]], dtype=np.float32).reshape(4,1)
            pts_proj, pts_in_cam_ref = project_pts_to_cam(p_xy_3d, cam_pos, K_homo)
            valid = pts_proj[0] > 0 and pts_proj[0] < self.width and pts_proj[1] > 0 and pts_proj[1] < self.height

        return p_xy_3d[:3].reshape(-1)

    # def check_in_view(self, u_min, u_max, v_min, v_max):
    #     valid = u_min < self.width and v_min < self.height and u_max > 0 and v_max > 0
    #     return valid