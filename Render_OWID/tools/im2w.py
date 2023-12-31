import numpy as np
import os
import json
import open3d as o3d
import imageio.v3 as iio
from tqdm import tqdm
import argparse

# path config
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--base_path', default = '/home/airlab/storage/airlab/data_0108/data/InsDet/OWID/P1', help='test config file path')
    parser.add_argument('--start', default = 0, type=int)
    parser.add_argument('--end', default = -1, type=int)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    base_path = args.base_path
    start = args.start
    end = args.end
    seqs = os.listdir(base_path)
    seqs.sort()
    for seq in tqdm(seqs[start:end]):
        seq_path = os.path.join(base_path, seq)
        if not os.path.isdir(seq_path):
            continue
        print("start seq: {}".format(seq))
        seq_path = os.path.join(base_path, seq)
        cam_path = os.path.join(seq_path, 'scene_camera.json')
        depth_path = os.path.join(seq_path, 'depth')
        mask_path = os.path.join(seq_path, 'mask')
        rgb_path = os.path.join(seq_path, 'rgb')
        info_path = os.path.join(seq_path, 'info.npz')
        point_path = os.path.join(seq_path, 'point')
        if os.path.isfile(info_path):
            print("finished seq: {}".format(seq))
            continue

        # camera par
        os.makedirs(point_path, exist_ok=True)
        with open (cam_path, 'r') as f:
            cam_para = json.load(f)

        # per image projection
        for img_id in cam_para.keys():
            # load image
            point_img_path = os.path.join(point_path, "{:06d}.npy".format(int(img_id)))
            depth_img_path = os.path.join(depth_path, "{:06d}.png".format(int(img_id)))
            rgb_img_path = os.path.join(rgb_path, "{:06d}.jpg".format(int(img_id)))
            mask_img_path = os.path.join(mask_path, "{:06d}.jpg".format(int(img_id)))
            depth_img = iio.imread(depth_img_path)
            rgb_img = iio.imread(rgb_img_path)
            # camera
            K = np.array(cam_para[img_id]["cam_K"]).reshape(3, 3)
            R = np.array(cam_para[img_id]["cam_R_w2c"]).reshape(3, 3)
            T = np.array(cam_para[img_id]["cam_t_w2c"]).reshape(3, 1)
            d_s = cam_para[img_id]["depth_scale"]
            depth_img = depth_img * d_s
            # visualize depth
            # depth_instensity = np.array(256 * depth_img / 0x0fff,
            #                         dtype=np.uint8)
            # iio.imwrite('grayscale_depth.png', depth_instensity)
            # depth 2 point in cam coordinate
            pcd_w = []
            colors = []
            point_img = np.zeros_like(rgb_img).astype(np.float64)
            height, width = depth_img.shape
            for i in range(height):
                for j in range(width):
                    z = depth_img[i][j]
                    x = (j - K[0][2]) * z / K[0][0]
                    y = (i - K[1][2]) * z / K[1][1]
                    p_cam = np.array([x, y, z]).reshape(3, 1)
                    p_w = np.linalg.inv(R) @ (p_cam - T)
                    point_img[i][j] = p_w.squeeze()
                    # pcd_w.append(list(p_w.squeeze()))
                    # colors.append(list(rgb_img[i][j] / 255))
            np.save(point_img_path, point_img)
        # # display pcd
    # pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd_w)  # set pcd_np as the point cloud points
    # pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    # # Visualize:
    # o3d.visualization.draw_geometries([pcd_o3d])

if __name__ == '__main__':
    main()

