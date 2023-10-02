import numpy as np
import os
import json
import open3d as o3d
import imageio.v3 as iio
from tqdm import tqdm

# path config
seq_path = '/home/airlab/storage/airlab/data_0108/data/InsDet/ZID-1M/P1/0'
cam_path = os.path.join(seq_path, 'scene_camera.json')
point_path = os.path.join(seq_path, 'point')
rgb_path = os.path.join(seq_path, 'rgb')
mask_path = os.path.join(seq_path, 'mask')

# camera par
# with open (cam_path, 'r') as f:
#     cam_para = json.load(f)

p_w = []
colors = []
data = np.load('/home/airlab/storage/airlab/data_0108/data/InsDet/ZID-1M/P1_point/30/info_point.npz')
# per image projection
for img_id in tqdm(range(data['rgb'].shape[0])):
    # data = np.load('output/info.npz')
    # load image
    # point_curr = np.load(os.path.join(point_path, "{:06d}.npy".format(int(img_id))))
    # mask_img_path = os.path.join(mask_path, "{:06d}.jpg".format(int(img_id)))
    point_curr = data['point'][img_id]
    rgb_img = data['rgb'][img_id]
    mask = data['mask'][img_id]
    # point_next = np.load(os.path.join(point_path, "{:06d}.npy".format(int(img_id)+1)))
    # rgb_img_path = os.path.join(rgb_path, "{:06d}.jpg".format(int(img_id)))
    # rgb_img = iio.imread(rgb_img_path)
    # mask = iio.imread(mask_img_path)
    height, width = point_curr.shape[1], point_curr.shape[2]
    h = np.arange(0, height)
    w = np.arange(0, width)

    # visualize
    for i in range(height):
        m = mask[:, i]
        if np.any(m):
            for j in range(width):
                if m[0][j]:
                    p_w.append(list(point_curr[:, i, j].astype(np.int64)))
                    colors.append(list(rgb_img[:, i, j]))

pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
pcd_o3d.points = o3d.utility.Vector3dVector(p_w)  # set pcd_np as the point cloud points
pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
# Visualize:
o3d.visualization.draw_geometries([pcd_o3d])

    # choose point
    # dis_list = []
    # one = np.ones((1, 1)).astype(np.float64)
    # while len(dis_list)<100:
    #     h_ind = np.random.choice(h, 1)[0]
    #     w_ind = np.random.choice(w, 1)[0]
    #     p_curr = point_curr[h_ind][w_ind]
    #     p_curr = np.concatenate([p_curr.reshape(3, 1), one], axis=0)
    #     p_next_cam = K @ np.concatenate([R, T], axis=-1) @ p_curr
    #     p_next_cam = p_next_cam.squeeze()
    #     p_next_cam /= p_next_cam[2]
    #     u, v = p_next_cam[0], p_next_cam[1]
    #     if 0<u<height and 0<v<width:
    #         p_next = point_next[int(u)][int(v)]
    #         dis = np.linalg.norm(p_next.reshape(3, 1)-p_curr[0:3])
    #         dis_list.append(dis)
    # avg_dis = np.mean(dis_list)
    # print("Geometry error between frame {:03d} and frame {:03d} is: {:04f}".format(int(img_id), int(img_id)+1, avg_dis))