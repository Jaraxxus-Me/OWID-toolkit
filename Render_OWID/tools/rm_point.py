import os

P1_path = '/home/airlab/storage/airlab/data_0108/data/InsDet/OWID/P1'

for ins in os.listdir(P1_path):
    full_path = os.path.join(P1_path, ins)
    if not os.path.isdir(full_path):
        continue
    os.remove(os.path.join(full_path, 'info.npz'))
    os.remove(os.path.join(full_path, 'info_dtoid.npz'))