import torch
import torch.nn as nn
from torch_pointnet_utils import PointNetSetAbstractionMsg
from models import chamfer_distance


class ComputeLoss3d(nn.Module):
    def __init__(self):
        super(ComputeLoss3d, self).__init__()

    def compute_chamfer_distance(self, p1, p2):
        '''
        Calculate Chamfer Distance between two point sets
        :param p1: size[bn, N, D]
        :param p2: size[bn, M, D]
        :return: sum of Chamfer Distance of two point sets
        '''

        diff = p1[:, :, None, :] - p2[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist_min1, _ = torch.min(dist, dim=2)
        dist_min2, _ = torch.min(dist, dim=1)

        return (torch.sum(dist_min1) + torch.sum(dist_min2)) / (p1.shape[0])

    def forward(self, gt_points, structure_points, origin_points=None):
        if origin_points is None:
            return self.compute_chamfer_distance(gt_points, structure_points)
        else:
            return self.compute_chamfer_distance(gt_points, structure_points) + self.compute_chamfer_distance(origin_points, structure_points)


class VecLoss(nn.Module):
    def __init__(self):
        super(VecLoss, self).__init__()
        self.vec_loss_fun = chamfer_distance.ComputeVecSimilarityLoss()

    def forward(self, structure_points, similarity_map, threshold=0.95):
        structure_points = structure_points.cuda()
        self.vec_loss = self.vec_loss_fun(structure_points, similarity_map, threshold) * 10
        return self.vec_loss


class WeightedChamferLoss(nn.Module):
    def __init__(self):
        super(WeightedChamferLoss, self).__init__()

    def compute_chamfer_distance(self, p1, p2):
        '''
        Calculate Chamfer Distance between two point sets
        :param p1: size[bn, N, D]
        :param p2: size[bn, M, D]
        :return: sum of Chamfer Distance of two point sets
        '''

        diff = p1[:, :, None, :] - p2[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist_min1, _ = torch.min(dist, dim=2)
        dist_min2, _ = torch.min(dist, dim=1)

        return (torch.mean(dist_min1) + torch.mean(dist_min2))

    def compute_end_distance(self, fps_points, structure_points, weight_map):
        '''
        Calculate Chamfer Distance between two point sets
        :param p1: size[bn, N, D]
        :param p2: size[bn, M, D]
        :return: sum of Chamfer Distance of two point sets
        '''
        # weight_map = torch.sigmoid(weight_map)
        weight_map = torch.sum(weight_map, dim=1)
        weight_map = torch.sigmoid(weight_map) + 1
        diff = structure_points[:, :, None, :] - fps_points[:, None, :, :]
        dist = torch.sum(diff * diff, dim=3)
        dist_min1, _ = torch.min(dist, dim=2)
        dist_min2, _ = torch.min(dist, dim=1)
        dist_min2 = dist_min2 * weight_map
        return (torch.mean(dist_min1) + torch.mean(dist_min2))

    def forward(self, fps_points, structure_points, weight_map, origin_points=None):
        if origin_points is None:
            return self.compute_end_distance(fps_points, structure_points, weight_map)
        else:
            return self.compute_end_distance(fps_points, structure_points, weight_map) + self.compute_chamfer_distance(origin_points, structure_points)


class Conv1dProbLayer(nn.Module):
    def __init__(self, in_channels, out_channels, out=False, kernel_size=1, dropout=0.2, normalize=False):
        super(Conv1dProbLayer, self).__init__()
        self.out = out
        self.dropout_conv_bn_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(num_features=out_channels),
        )
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        self.normalize = normalize
        self.normalize_layer = Normalize(dim=1)

    def forward(self, x):
        x = self.dropout_conv_bn_layer(x)
        if self.normalize:
            x = self.normalize_layer(x)
        if self.out:
            x = self.softmax(x)
        else:
            x = self.relu(x)
        return x


class RefineNet(nn.Module):
    def __init__(self, num_structure_points, in_channel=128 + 256 + 256, out=True, normalize=False):
        super(RefineNet, self).__init__()

        conv1d_stpts_prob_modules = []
        if num_structure_points <= in_channel:
            conv1d_stpts_prob_modules.append(
                Conv1dProbLayer(in_channels=in_channel, out_channels=512, kernel_size=1, out=False))
            in_channels = 512
            while in_channels >= num_structure_points * 2:
                out_channels = int(in_channels / 2)
                conv1d_stpts_prob_modules.append(
                    Conv1dProbLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, out=False))
                in_channels = out_channels
            conv1d_stpts_prob_modules.append(
                Conv1dProbLayer(in_channels=in_channels, out_channels=num_structure_points, kernel_size=1, out=out,
                                normalize=normalize))
        else:
            conv1d_stpts_prob_modules.append(
                Conv1dProbLayer(in_channels=in_channel, out_channels=1024, kernel_size=1, out=False))
            in_channels = 1024
            while in_channels <= num_structure_points / 2:
                out_channels = int(in_channels * 2)
                conv1d_stpts_prob_modules.append(
                    Conv1dProbLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, out=False))
                in_channels = out_channels
            conv1d_stpts_prob_modules.append(
                Conv1dProbLayer(in_channels=in_channels, out_channels=num_structure_points, kernel_size=1, out=out,
                                normalize=normalize))

        self.conv1d_stpts_prob = nn.Sequential(*conv1d_stpts_prob_modules)

    def forward(self, features):
        return self.conv1d_stpts_prob(features)


class OffsetNet(nn.Module):
    def __init__(self, num_structure_points=16, in_channel=128 + 256 + 256, mlp_list=[512, 256]):
        super(OffsetNet, self).__init__()
        self.Offset_modules = nn.ModuleList()
        self.num_structure_points = num_structure_points
        Offset_modules = []
        last_channel = in_channel
        for out_channel in mlp_list:
            Offset_modules.append(nn.Conv1d(last_channel, out_channel, 1))
            Offset_modules.append(nn.BatchNorm1d(out_channel))
            Offset_modules.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.Offset_modules = nn.Sequential(*Offset_modules)
        self.reshape_module = nn.Conv1d(last_channel, 6, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        # batch_size, in_channel, points_num = features.shape
        # print(features.shape)
        # features = features.permute(0, 2, 3, 1)
        # print(features.shape)
        features = self.Offset_modules(features)
        # features = torch.max(features, dim=1, keepdim=False)[0]
        features = self.reshape_module(features)
        return features.permute(0, 2, 1)


class FeatureMergeBlock(nn.Module):
    def __init__(self, num_structure_points, boom_rate=2):
        super(FeatureMergeBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.merge_layer = nn.Sequential(
            nn.Conv2d(num_structure_points, boom_rate * num_structure_points, kernel_size=1),
            nn.BatchNorm2d(boom_rate * num_structure_points),
            nn.ReLU(inplace=True),
            nn.Conv2d(boom_rate * num_structure_points, num_structure_points, kernel_size=1),
            nn.BatchNorm2d(num_structure_points),
        )

    def forward(self, features):
        return self.relu(self.merge_layer(features) + features)
        # return self.merge_layer(features) + features


class Normalize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        return x / norm


class Pointnet2StructurePointNet(nn.Module):

    def __init__(self, num_structure_points, input_channels=3, multi_distribution_num=1, offset=False,
                 merge_block_num=1):
        super(Pointnet2StructurePointNet, self).__init__()
        self.point_dim = 3
        self.num_structure_points = num_structure_points
        self.offset = offset
        self.input_channels = input_channels
        self.num_structure_points = num_structure_points

        self.sa1 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 128],
                                             in_channel=0, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.2, 0.4, 0.8], nsample_list=[32, 64, 128],
                                             in_channel=64 + 128 + 128,
                                             mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])

        self.multi_distri_layers = nn.ModuleList()
        for i in range(multi_distribution_num):
            self.multi_distri_layers.append(RefineNet(num_structure_points, in_channel=128 + 256 + 256, out=False))

        feature_merge = []
        for _ in range(merge_block_num):
            feature_merge.append(FeatureMergeBlock(num_structure_points=num_structure_points))
        self.feature_merge = nn.Sequential(*feature_merge)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, pointcloud):
        '''
        :param pointcloud: input point cloud with shape (bn, num_of_pts, 3)
        :param return_weighted_feature: whether return features for the structure points or not
        :return:
        '''
        _ = None
        B = pointcloud.shape[0]
        if pointcloud.shape[2] == 3:
            pointcloud = pointcloud.permute(0, 2, 1)

        if pointcloud.shape[1] > 3:
            xyz = pointcloud[:, :3, :]
            features = pointcloud[:, 3:, :]
        else:
            xyz = pointcloud
            features = None
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)

        stpts_prob_map = []
        for i in range(len(self.multi_distri_layers)):
            stpts_prob_map.append(self.multi_distri_layers[i](features))

        stpts_prob_map = torch.stack(stpts_prob_map, dim=3)
        stpts_prob_map = torch.max(self.feature_merge(stpts_prob_map), dim=3)[0]
        stpts_prob_map = self.softmax(stpts_prob_map)

        # (4,16,128) *(4,3,128)
        if not xyz.shape[2] == 3:
            xyz = xyz.permute(0, 2, 1)

        structure_points = torch.sum(stpts_prob_map[:, :, :, None] * xyz[:, None, :, :], dim=2)
        # features = features * torch.sum(stpts_prob_map, dim=1, keepdim=True)
        # weighted_features = torch.sum(stpts_prob_map[:, None, :, :] * features[:, :, None, :], dim=3)
        # cos_similarity = SimilarityPoints(weighted_features)
        cos_similarity = None
        return structure_points, xyz, cos_similarity, stpts_prob_map


if __name__ == '__main__':
    data = torch.rand(2, 3, 1024)
    print("===> testing pointSPN ...")
    model = Pointnet2StructurePointNet(num_structure_points=16, offset=False)
    data = data.cuda()
    model = model.cuda()
    structure_points, xyz, cos_similarity, stpts_prob_map = model(data)
    loss = ComputeLoss3d()
    loss = loss.cuda()
    print(loss(xyz, structure_points))
