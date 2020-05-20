import cv2

import sys
sys.path.append("../")
from cvo_ops.utils_general.io import read_calib_file


def show_kpt(fid):
    img_path = '/media/sda1/datasets/extracted/shapenet_render/syn_rgb/car/blender-{:06d}.color.png'.format(fid)
    txt_path = '/media/sda1/datasets/extracted/shapenet_render/syn_rgb/car/blender-{:06d}.color.txt'.format(fid)

    img = cv2.imread(img_path)
    result = read_calib_file(txt_path)

    pts = result['pts_proj']
    pts = pts.reshape(-1, 3)
    
    bbox = result["bbox_uuvv"]
    bbox = [int(round(b)) for b in bbox]

    cv2.rectangle(img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (255, 0, 0), 1)

    for i in range(pts.shape[0]):
        cv2.circle(img, (int(round(pts[i,0])), int(round(pts[i,1]))), radius=1, color=(0,0,255))

    # cv2.line(img, (0,0), (img.shape[1]-1, img.shape[0]-1), color=(0,0,255))
    # cv2.line(img, (img.shape[1]-1,0), (0, img.shape[0]-1), color=(0,0,255))
    # with open(txt_path) as f:
    #     lines = f.readlines()

    # pts = []
    # for i, line in enumerate(lines):
    #     pt = [int(round(float(x))) for x in line.split(' ')]
    #     pts.append(pt)

    #     # cv2.circle(img, tuple([pt[1], pt[0]]), radius=5, color=(0,0,255))
    #     cv2.circle(img, tuple(pt[:2]), radius=5, color=(0,0,255))

    cv2.imshow('img{}'.format(fid), img)
    cv2.waitKey(0)

if __name__ == "__main__":
    for fid in range(1, 5):
        show_kpt(fid)