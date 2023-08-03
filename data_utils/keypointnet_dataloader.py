import torch.utils.data as data
from .ModelNetDataLoader import pc_normalize
import json
import numpy as np


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=float)
    return pc


class KeyPointNetDataLoader(data.Dataset):
    def __init__(self, num_points, json_path, pcd_path, split='train'):
        super().__init__()
        self.num_points = num_points
        self.data_path = json.load(open(json_path))
        self.pcd_path = pcd_path
        if split == 'train':
            self.data_path = self.data_path[:-200]
        else:
            self.data_path = self.data_path[-200:]

    def __getitem__(self, idx):
        # res = {}
        class_id = self.data_path[idx]['class_id']
        model_id = self.data_path[idx]['model_id']
        points = naive_read_pcd(r'{}/{}/{}.pcd'.format(self.pcd_path, class_id, model_id)).astype(np.float32)
        points = pc_normalize(points)
        ground_truths = np.array(
            [points[point['pcd_info']['point_index']] for point in self.data_path[idx]['keypoints']])
        ground_truths_num = ground_truths.shape[0]
        ground_truths = np.pad(ground_truths, ((0, 18 - ground_truths_num), (0, 0,)), 'constant',
                               constant_values=(0, 0))
        return points[:self.num_points, :], ground_truths, ground_truths_num, ground_truths_num

    def __len__(self):
        return len(self.data_path)


def paint(points_xyz, origin_xyz):
    import matplotlib.pyplot as plt

    x1 = points_xyz[:, 0]
    y1 = points_xyz[:, 1]
    z1 = points_xyz[:, 2]

    x2 = origin_xyz[:, 0]
    y2 = origin_xyz[:, 1]
    z2 = origin_xyz[:, 2]

    ax1 = plt.subplot(111, projection='3d')
    ax1.scatter(x1, y1, z1, c=COLOR_LIST[:points_xyz.shape[0], :] / 255, s=48)
    ax1.scatter(x2, y2, z2, c='#A9A9A9', s=1)
    ax1.axis('off')
    plt.show()


def create_color_list(num):
    import random
    colors = np.ndarray(shape=(num, 3))
    for i in range(0, num):
        colors[i, 0] = random.randint(0, 255)
        colors[i, 1] = random.randint(0, 255)
        colors[i, 2] = random.randint(100, 255)

    colors[0, :] = np.array([0, 0, 0]).astype(int)
    colors[1, :] = np.array([146, 61, 10]).astype(int)
    colors[2, :] = np.array([102, 97, 0]).astype(int)
    colors[3, :] = np.array([255, 0, 0]).astype(int)
    colors[4, :] = np.array([113, 0, 17]).astype(int)
    colors[5, :] = np.array([255, 127, 39]).astype(int)
    colors[6, :] = np.array([255, 242, 0]).astype(int)
    colors[7, :] = np.array([0, 255, 0]).astype(int)
    colors[8, :] = np.array([0, 0, 255]).astype(int)
    colors[9, :] = np.array([15, 77, 33]).astype(int)
    colors[10, :] = np.array([163, 73, 164]).astype(int)
    colors[11, :] = np.array([255, 174, 201]).astype(int)
    colors[12, :] = np.array([255, 220, 14]).astype(int)
    colors[13, :] = np.array([181, 230, 29]).astype(int)
    colors[14, :] = np.array([153, 217, 234]).astype(int)
    colors[15, :] = np.array([112, 146, 190]).astype(int)

    return colors


COLOR_LIST = create_color_list(200)

if __name__ == '__main__':
    import torch

    data = KeyPointNetDataLoader(json_path='./keypointnet/annotations/mug.json', pcd_path='../keypointnet/pcds',
                                 split='val',num_points=1024)

    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True, num_workers=10, drop_last=True)
    print(len(DataLoader))
    for point, label, ground_truths_num in DataLoader:
        paint(label[0], point[0])
        # print(point.shape)
        # print(label.shape)
        # print(ground_truths_num)
