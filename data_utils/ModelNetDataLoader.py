'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import argparse
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, category='chair', split='train', process_data=False):
        self.root = root
        self.npoints = args.num_inputs
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.category = category
        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]

        if not category == 'all':
            self.datapath = [
                (shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                in range(len(shape_ids[split])) if shape_names[i] == category]
        else:
            self.datapath = [
                (shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root,
                                          'modelnet%d_%s_%d_%spts_fps.dat' % (self.num_category, split, self.npoints, self.category))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%d_%spts.dat' % (self.num_category, split, self.npoints, self.category))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-batch_size", type=int, default=20, help="Batch size")
    parser.add_argument(
        "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff"
    )
    parser.add_argument("-lr", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_batch", type=float, default=20, help="Learning rate decay batch"
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"
    )

    parser.add_argument(
        "-checkpoint_save_step", type=int, default=50, help="Step for saving Checkpoint"
    )

    parser.add_argument(
        "-checkpoint", type=str, default=None
        , help="Checkpoint to start from"
    )
    parser.add_argument(
        "-num_of_transform", type=int, default=0,
        help="Number of transforms for rotation data augmentation. Useful when testing on shapes without alignment"
    )

    parser.add_argument(
        "-num_inputs", type=int, default=1024, help="sample points from initial point cloud"
    )

    parser.add_argument(
        "-num_structure_points", type=int, default=16
        , help="Number of structure points"
    )
    parser.add_argument(
        "-category", type=str, default='chair', help="Category of the objects to train"
    )
    parser.add_argument(
        "-data_dir", type=str, default="training_data/", help="Root of the training data"
    )
    parser.add_argument(
        "-test_data_dir", type=str, default="demo_data/", help="Root of the test data"
    )
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')

    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    parser.add_argument(
        "-max_epochs", type=int, default=200, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-log_dir", type=str, default=None, help="Root of the log"
    )
    parser.add_argument(
        "-multi_distribution", type=int, default=5, help="Multivariate normal distribution nums"
    )
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('-model', default='PointSPN', help='model name [default: PointSPN]')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import torch

    args = parse_args()
    data = ModelNetDataLoader('../modelnet40/', args, split='train')

    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True, num_workers=10)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
