import os
from re import L
from tqdm import tqdm
import json
import shutil

original_p1_path = "/home/airlab/storage/airlab/OWID/P1_raw_abo"
p1_path = "/home/airlab/storage/airlab/data_0108/data/InsDet/OWID/P1"
name_id_mapper = "/home/airlab/storage/airlab/data_0108/data/InsDet/OWID/P1/ZID-10k_mapper.json"
name_id_mapper_new = "/home/airlab/storage/airlab/data_0108/data/InsDet/OWID/P1/ZID-10k_mapper_new.json"

with open(name_id_mapper, 'r') as f:
    mapper = json.load(f)
obj_ids = [i for i in list(mapper.values())]
obj_ids.sort()
obj_id = obj_ids[-1] + 1

for syth_id in tqdm(os.listdir(original_p1_path)):
    for source_id in os.listdir(os.path.join(original_p1_path, syth_id)):
        mapper["{}_{}".format(syth_id, source_id[:-3])] = obj_id
        assert obj_id not in obj_ids
        obj_ids.append(obj_id)
        # os.makedirs(os.path.join(p1_path, str(obj_id)), exist_ok=True)

        temp = os.listdir(os.path.join(original_p1_path, syth_id, source_id, 'test_video'))[0]
        source_dir = os.path.join(original_p1_path, syth_id, source_id, 'test_video', temp)
        shutil.copytree(source_dir, os.path.join(p1_path, str(obj_id)), dirs_exist_ok=True)
        print("Copied: {:06d}".format(obj_id))
        obj_id += 1
print('Total instances: {}'.format(len(obj_ids)))
with open(name_id_mapper_new, 'w') as f:
    json.dump(mapper, f)