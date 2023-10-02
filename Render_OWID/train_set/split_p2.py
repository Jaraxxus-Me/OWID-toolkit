import os
from re import L
from threading import local
from tqdm import tqdm
import json
import shutil
from pycocotools.coco import COCO
import numpy as np

original_p2_anno = "/home/airlab/storage/airlab/data_0108/data/InsDet/OWID/P2/coco_annotations.json"
coco = COCO(original_p2_anno)
train_ratio = 0.998

with open(original_p2_anno, 'r') as f:
    anno_dict = json.load(f)
all_obj_id = list(range(len(anno_dict['categories'])))
train_obj_id = np.random.choice(all_obj_id, size = round(len(all_obj_id)*train_ratio), replace=False)
val_obj_id = [i for i in all_obj_id if i not in train_obj_id]

train_anno = anno_dict.copy()
val_anno = anno_dict.copy()

train_anno["images"] = []
train_anno["annotations"] = []
val_anno["images"] = []
val_anno["annotations"] = []
train_im_id = 0
val_im_id = 0
train_ann_id = 0
val_ann_id = 0

for im in tqdm(coco.imgs.keys()):
    local_image_info = coco.imgs[im]
    anns = coco.loadAnns(ids=coco.getAnnIds(im))
    train_flag = True
    val_flag = False
    for ann in anns:
        if ann['category_id'] in val_obj_id:
            ann['id'] = val_ann_id
            ann['image_id'] = val_im_id
            val_anno["annotations"].append(ann)
            val_ann_id +=1
            val_flag = True
            train_flag = False
    if val_flag:
        local_image_info['id'] = val_im_id
        val_anno["images"].append(local_image_info)
        val_im_id += 1
    else:
        assert train_flag
        for ann in anns:
            assert ann['category_id'] in train_obj_id
            ann['id'] = train_ann_id
            ann['image_id'] = train_im_id
            train_anno["annotations"].append(ann)
            train_ann_id +=1
        local_image_info['id'] = train_im_id
        train_anno["images"].append(local_image_info)
        train_im_id += 1

with open(os.path.join("/home/airlab/storage/airlab/data_0108/data/InsDet/OWID/P2/", 'train_annotations.json'), 'w') as f:
    json.dump(train_anno, f)
print('Total train images: {}'.format(train_im_id))
print('Total train instances: {}'.format(train_ann_id))

with open(os.path.join("/home/airlab/storage/airlab/data_0108/data/InsDet/OWID/P2/", 'val_annotations.json'), 'w') as f:
    json.dump(val_anno, f)
print('Total val images: {}'.format(val_im_id))
print('Total val instances: {}'.format(val_ann_id))
