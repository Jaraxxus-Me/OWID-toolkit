import os
from mmdet.datasets.pipelines import Compose
from tqdm import tqdm
import numpy as np
import shutil

P1_base = '/home/user/storage/airlab/data_0108/data/InsDet/OWID/P1'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# need to coment out the Collect function
P1_pipeline = [
    dict(type='LoadP1Info', target_sz=128),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImagesToTensor', keys=['rgb', 'mask']),
    dict(type='Collect', keys=['rgb', 'mask']),
]

pipeline = Compose(P1_pipeline)
obj_ids = os.listdir(P1_base)

for obj_id in tqdm(obj_ids):
    base_path = os.path.join(P1_base, '{}'.format(obj_id))
    if os.path.isdir(base_path):
        print("Start: {}".format(obj_id))
        if os.path.isfile(os.path.join(base_path, 'info.npz')):
            print("Finished: {}".format(obj_id))
            continue
        results = dict(P1_path=P1_base, obj_id=obj_id)
        data = pipeline(results)
        np.savez(os.path.join(base_path, 'info.npz'), rgb=data['rgb'], mask=data['mask'])