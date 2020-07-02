
from coco import COCO

def get_img_ids_from_cat_name(cocoins, cat_name_list):
    cat_ids = cocoins.getCatIds(supNms=cat_name_list)
    img_ids = cocoins.getImgIds(catIds=cat_ids)
    return img_ids

if __name__ == "__main__":
    ### load background images info
    # coco_val2017 = COCO(annotation_file="/media/sda1/datasets/extracted/COCO/annotations/instances_val2017.json")
    coco_val2017 = COCO(annotation_file="/media/sda1/datasets/extracted/COCO/annotations/person_keypoints_val2017.json")

    print(coco_val2017.dataset['categories'])

    # veh_name_list = ["vehicle"]
    # veh_img_ids = get_img_ids_from_cat_name(coco_val2017, veh_name_list)
    # all_img_ids = coco_val2017.getImgIds()

    # ### example to obtain the categories of an image
    # img_id = 347544
    # ann_id = coco_val2017.getAnnIds(imgIds=img_id)
    # anns = coco_val2017.loadAnns(ids=ann_id)
    # cat_id = [int(a["category_id"]) for a in anns]
    # cat = coco_val2017.loadCats(ids=cat_id)
    # print(cat)
    # print(img_id in veh_img_ids)