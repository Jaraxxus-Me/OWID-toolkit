import os
from re import L
from threading import local
from tqdm import tqdm
import json
import shutil
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

original_p2_path = "/home/airlab/storage/airlab/OWID/P2_raw_2_abo"
p2_path = "/home/airlab/storage/airlab/data_0108/data/InsDet/OWID/P2_aug"
name_id_mapper = "/home/airlab/storage/airlab/data_0108/data/InsDet/OWID/P1/ZID-10k_mapper_new.json"

with open(name_id_mapper, 'r') as f:
    mapper = json.load(f)
coco_anno = {"info": {"description": "zid10k_coco_annotations", 
"url": "https://github.com/waspinator/pycococreator", 
"version": "0.1.0", 
"year": 2022, "contributor": "Unknown", "date_created": "2022-10-15 18:58:55.838053"}, 
"licenses": [{"id": 1, "name": "Attribution-NonCommercial-ShareAlike License", "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"}],
"categories": [],
"images": [],
"annotations": []}
for name_key in mapper:
    class_dict = {'id': mapper[name_key],
    "supercategory": "owid-instances",
    "name": "ins_{:06d}".format(mapper[name_key])}
    coco_anno["categories"].append(class_dict)

image_id = 0
ins_id = 1
box_area = []
num_inst = []

for syth_id in tqdm(os.listdir(original_p2_path)):
    local_coco = COCO(os.path.join(original_p2_path, syth_id, 'coco_annotations.json'))
    with open(os.path.join(original_p2_path, syth_id, 'orientation.pkl'), 'rb') as f:
        local_orientation = pkl.load(f)

    for local_im in local_coco.imgs.keys():
        # copy local images to world images
        local_image_info = local_coco.imgs[local_im]
        local_image_info["id"] = image_id
        img_source_dir = os.path.join(original_p2_path, syth_id, local_image_info["file_name"])
        local_image_info["file_name"] = "images/{:06d}.jpg".format(image_id)
        img_target_dir = os.path.join(p2_path, local_image_info["file_name"])

        shutil.copy(img_source_dir, img_target_dir)
        coco_anno["images"].append(local_image_info)

        # os.makedirs(os.path.join(p1_path, str(obj_id)), exist_ok=True)
        local_anno_info = local_coco.imgToAnns[local_im]
        local_ins_num = 0
        # negtive boxes
        for ann_i in local_anno_info:
            if ann_i['category_id'] == -1:
                continue
            local_ins_num += 1

        for ann in local_anno_info:
            if ann['category_id'] == -1:
                continue
            ann['id'] = ins_id
            ann['neg_box'] = []
            # negtive boxes
            for ann_n in local_anno_info:
                if (ann_n['category_id'] != ann['category_id']) and (ann_n['category_id'] != -1):
                    ann['neg_box'].append(ann_n['bbox'])
            ann['image_id'] = image_id
            ann['num_ins'] = local_ins_num
            ann['cam_w'] = local_orientation['camera_w'].tolist()
            ann['obj_w'] = local_orientation['{}_w'.format(ann['category_id'])].tolist()
            num_inst.append(local_ins_num)
            box_area.append(ann['area'])
            coco_anno["annotations"].append(ann)
            ins_id += 1
        print("Copied: {:06d}".format(image_id))
        image_id += 1
with open(os.path.join(p2_path, 'coco_annotations.json'), 'w') as f:
    mapper = json.dump(coco_anno, f)
print('Total images: {}'.format(image_id))
print('Total instances: {}'.format(ins_id))
# the histogram of the inst num
n, bins, patches = plt.hist(num_inst, 5, density=True, facecolor='g', alpha=0.75)


plt.xlabel('Ins. Num')
plt.ylabel('Fre')
plt.title('Histogram of Ins. Num')
plt.grid(True)
plt.show()

plt.savefig('ins_num_0308_num.png')

# the histogram of the box area
n, bins, patches = plt.hist(box_area, 50, density=True, facecolor='g', alpha=0.75)


plt.xlabel('Box. Area')
plt.ylabel('Fre')
plt.title('Histogram of Box. Area')
plt.grid(True)
plt.show()

plt.savefig('ins_num_0130_area.png')