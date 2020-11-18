"""This file is to generate binary mask image from json file containing polygon annotations"""
### the annotation tool is https://www.makesense.ai/ 
import numpy as np
import cv2
import json
### https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python
# ### jackson left-turn intersection
# json_path = "/home/minghanz/Pictures/empty_road/labels_road_20200614122506.json"
# image_name = "vlcsnap-2020-06-05-00h54m01s511.png"
# img_shape = (480,852)

### Ko-PER dataset
json_path = "/media/sda1/datasets/extracted/KoPER/added/labels_road_mask_polygon_20200701091947.json"
image_name = "SK_4_empty_road.png"
img_shape = (494,656)

with open(json_path) as f:
    mask_info = json.load(f)
    print(mask_info)

pts_x = mask_info[image_name]["regions"]["0"]['shape_attributes']["all_points_x"]
pts_y = mask_info[image_name]["regions"]["0"]['shape_attributes']["all_points_y"]

contours = np.array([pts_x, pts_y]).T 
contours = contours.round().astype(int)
print(contours.shape)

# contours = np.array( [ [50,50], [50,150], [150, 150], [150,50] ] )
img = np.zeros( img_shape ) # create a single channel 200x200 pixel black image 
cv2.fillPoly(img, pts =[contours], color=(255,255,255))
cv2.imshow(" ", img)
cv2.waitKey()

cv2.imwrite("road_mask.png", img)