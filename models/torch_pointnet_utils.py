import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import matplotlib.pyplot as plt

# import open3d as o3d

from pointnet2_ops import pointnet2_utils


# from PointSPN_Plus import Conv1dProbLayer,RefineNet

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    group_idx[group_idx == N] = 0
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # fps_idx = z_order.z_order_point_sample(xyz, npoint)  # [B, npoint, C]
    # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
    # fps_idx = farthest_point_sample(xyz, self.groups).long()
    try:
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long()  # [B, npoint]
    except:
        fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        # new_xyz = index_points(xyz, z_order.z_order_point_sample(xyz, S))
        # new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        xyz = xyz.contiguous()
        new_xyz = index_points(xyz, pointnet2_utils.furthest_point_sample(xyz, S).long())

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: sampled input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointFPSBlock(nn.Module):
    def __init__(self, fps_points, in_channel):
        super(PointFPSBlock, self).__init__()
        self.refine_module = FPSRefineNet(fps_points, in_channel=in_channel, out=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, grouped_xyz, grouped_points):
        """
        Input:
            grouped_xyz: input points position data, [B, C, K, S]
            grouped_points: input points data, [B, D, K, S]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        grouped_points = self.refine_module(grouped_points)
        grouped_points = self.softmax(grouped_points)
        offset_xyz = torch.sum(grouped_points[:, None, :, :, :] * grouped_xyz[:, :, None, :, :], dim=3)
        return offset_xyz


class FPSConv2dProbLayer(nn.Module):
    def __init__(self, in_channels, out_channels, out=False, kernel_size=1, dropout=0.2):
        super(FPSConv2dProbLayer, self).__init__()
        self.out = out
        self.dropout_conv_bn_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.dropout_conv_bn_layer(x)
        if self.out:
            x = self.softmax(x)
        else:
            x = self.relu(x)
        return x


class FPSRefineNet(nn.Module):
    def __init__(self, num_structure_points, in_channel=128 + 256 + 256, out=True):
        super(FPSRefineNet, self).__init__()

        Conv2d_stpts_prob_modules = []
        if num_structure_points <= in_channel:
            Conv2d_stpts_prob_modules.append(
                FPSConv2dProbLayer(in_channels=in_channel, out_channels=512, kernel_size=1, out=False))
            in_channels = 512
            while in_channels >= num_structure_points * 2:
                out_channels = int(in_channels / 2)
                Conv2d_stpts_prob_modules.append(
                    FPSConv2dProbLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, out=False))
                in_channels = out_channels
            Conv2d_stpts_prob_modules.append(
                FPSConv2dProbLayer(in_channels=in_channels, out_channels=num_structure_points, kernel_size=1, out=out))
        else:
            Conv2d_stpts_prob_modules.append(
                FPSConv2dProbLayer(in_channels=in_channel, out_channels=1024, kernel_size=1, out=False))
            in_channels = 1024
            while in_channels <= self.num_structure_points / 2:
                out_channels = int(in_channels * 2)
                Conv2d_stpts_prob_modules.append(
                    FPSConv2dProbLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, out=False))
                in_channels = out_channels
            Conv2d_stpts_prob_modules.append(
                FPSConv2dProbLayer(in_channels=in_channels, out_channels=num_structure_points, kernel_size=1, out=out))

        self.Conv2d_stpts_prob = nn.Sequential(*Conv2d_stpts_prob_modules)

    def forward(self, features):
        return self.Conv2d_stpts_prob(features)


class FPSPointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(FPSPointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
        self.fps_block = PointFPSBlock(fps_points=1, in_channel=in_channel + 3)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint

        # new_xyz = index_points(xyz, z_order.z_order_point_sample(xyz, S))
        try:
            new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        except:
            new_xyz = index_points(xyz, pointnet2_utils.furthest_point_sample(xyz, S).long())

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            fps_points_offset = self.fps_block(grouped_xyz.permute(0, 3, 2, 1), grouped_points).view(B, C, S)

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, fps_points_offset, new_points_concat


class StructureSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(StructureSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, structure_xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        B, N, C = xyz.shape
        S = self.npoint

        # new_xyz = index_points(xyz, z_order.z_order_point_sample(xyz, S))
        # new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_xyz = structure_xyz
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)

        cos_similarity = torch.zeros(B, S, S, dtype=torch.float32)
        for i in range(S):
            cos_similarity[:, i, :] = F.cosine_similarity(new_points_concat[:, :, i].view(B, -1, 1), new_points_concat,
                                                          dim=1)
            # print(F.cosine_similarity(new_points_concat[:, :, i].view(B, -1, 1), new_points_concat, dim=1).shape)

        return cos_similarity


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouperKnn(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list):
        super(LocalGrouperKnn, self).__init__()
        self.npoint = npoint
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(nsample_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points, new_xyz):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint

        # new_xyz = index_points(xyz, z_order.z_order_point_sample(xyz, S))
        # new_xyz = index_points(xyz, farthest_point_sample(xyz, S))

        new_points_list = []
        for i, K in enumerate(self.nsample_list):
            # group_idx = query_ball_point(radius, K, xyz, new_xyz)
            # grouped_xyz = index_points(xyz, group_idx)
            group_idx = knn_point(self.kneighbors, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)  # [B, npoint, k, 3]
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            grouped_points = grouped_xyz

            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]
            mean = torch.cat([grouped_points, new_xyz], dim=-1)

            mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, mlp_list, use_xyz=True):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_channel = channel + 3
        for out_channel in mlp_list:
            self.convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points, new_xyz):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]
        if points is not None:
            points = points.permute(0, 2, 1)
        new_points = new_xyz
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]

        grouped_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)],
                                   dim=-1)
        grouped_points = grouped_points.permute(0, 3, 2, 1)
        for j in range(len(self.convs)):
            conv = self.convs[j]
            bn = self.bns[j]
            grouped_points = F.relu(bn(conv(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]

        return new_xyz, new_points


def plot_Matrix(cm):
    cmap = plt.cm.Reds
    # classes = ['Social', 'Using computer', 'Reading']  # 规定出来的x,y轴的值，
    classes = range(cm.shape[1])
    plt.rc('font', size='10')  # 设置字体大小

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),  # 设置坐标轴显示的样式
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Matrix',
           ylabel='Point',
           xlabel='Point')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)

    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.

    cm = np.around(cm, 2)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(i, j, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.show()


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def SimilarityPoints(points):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        structure_xyz: query points, [B, S, C]
        points:all points feature, [B, D, N]

    """
    B, C, N = points.shape

    cos_similarity = torch.zeros(B, N, N, dtype=torch.float32)
    for i in range(N):
        cos_similarity[:, i, :] = F.cosine_similarity(points[:, :, i].view(B, -1, 1), points, dim=1)
        # cos_similarity[:, i, :] = torch.dist(grouped_xyz_features[:, :, i].view(B, -1, 1), grouped_xyz_features, p=1)
    return cos_similarity


def SimilarityKnn(xyz, structure_xyz, points):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        structure_xyz: query points, [B, S, C]
        points:all points feature, [B, D, N]

    """
    B, N, C = xyz.shape
    _, S, D = structure_xyz.shape
    idx = knn_point(1, xyz, structure_xyz)
    grouped_xyz_features = index_points(points.permute(0, 2, 1), idx).view(B, S, -1).permute(0, 2, 1)
    cos_similarity = torch.zeros(B, S, S, dtype=torch.float32)
    for i in range(S):
        cos_similarity[:, i, :] = F.cosine_similarity(grouped_xyz_features[:, :, i].view(B, -1, 1),
                                                      grouped_xyz_features, dim=1)
        # cos_similarity[:, i, :] = torch.dist(grouped_xyz_features[:, :, i].view(B, -1, 1), grouped_xyz_features, p=1)
    return cos_similarity


# def SimilarityO3d(points):
#     """
#     Input:
#         nsample: max sample number in local region
#         xyz: all points, [B, N, C]
#         structure_xyz: query points, [B, S, C]
#         points:all points feature, [B, D, N]
#
#     """
#     B, C, S = points.shape
#
#     cos_similarity = torch.zeros(B, S, S, dtype=torch.float32)
#     for i in range(S):
#         cos_similarity[:, i, :] = F.cosine_similarity(points[:, :, i].view(B, -1, 1), points, dim=1)
#         # cos_similarity[:, i, :] = torch.dist(grouped_xyz_features[:, :, i].view(B, -1, 1), grouped_xyz_features, p=1)
#     return cos_similarity
#
#
# def o3d_feature(xyz, xyz_1):
#     B, N, C = xyz.shape
#     _, M, _ = xyz_1.shape
#     xyz = torch.cat([xyz_1, xyz], dim=1)
#     xyz_feature = torch.empty((B, 33, N + M))
#     for i in range(B):
#         point_cloud_data = xyz[i].cpu().detach().numpy()
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
#         pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, 0.5)
#         xyz_feature[i, :, :] = torch.from_numpy(np.array(pcd_fpfh.data))
#     return xyz_feature[:, :, :M]
#
#
# def preprocess_point_cloud(pcd, voxel_size):
#     # print(":: Downsample with a voxel size %.3f." % voxel_size)
#     # pcd_down = pcd.voxel_down_sample(voxel_size)
#     pcd_down = pcd
#
#     radius_normal = 0.2
#     # print(":: Estimate normal with search radius %.3f." % radius_normal)
#     pcd_down.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=20))
#
#     radius_feature = 0.2
#     # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
#
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=20))
#
#     return pcd_down, pcd_fpfh
#
#
# def cluster(xyz):
#     # print("->正在DBSCAN聚类...")
#     eps = 0.2  # 同一聚类中最大点间距
#     min_points = 10  # 有效聚类的最小点数
#
#     for i in range(xyz.shape[0]):
#         point_cloud_data = xyz[i].cpu().detach().numpy()
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
#         with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#             labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=True))
#         max_label = labels.max()  # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
#         # print(f"point cloud has {max_label + 1} clusters")
#         # print(f"point cloud has {max_label + 1} clusters")
#         colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
#         colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
#         pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
#         o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    # print("===> testing pointSPN ...")
    data = torch.rand(2, 128, 3)
    data_1 = torch.rand(2, 16, 3)
    data_2 = torch.rand(2, 256, 128)

    model = LocalGrouper(channel=256 + 3, groups=16, kneighbors=20, mlp_list=[128, 128, 256], use_xyz=True)
    model(data, data_2, data_1)
    # cluster(data_1)
    # feature = o3d_feature(data, data_1)
    # Similarity = SimilarityO3d(feature)
    # plot_Matrix(Similarity[0].detach().numpy())
    # print(o3d_feature(data, data_1).shape)
    # data_1 = torch.rand(2, 16, 3)
    # data_2 = torch.rand(2, 256, 128)
    # model = LocalGrouperKnn(npoint=16, nsample_list=[[16]], in_channel=3, mlp_list=[])
    # model(data, data_2, data_1)

    # print(SimilarityKnn(data, data_1, data_2).shape)
    # data_2 = torch.rand(2, 128, 128)

    # print("===> testing pointSPN ...")
    # model = PointFPSBlock(fps_points=1, in_channel=12)
    # out = model(data, data_1)
    # print(out.shape)
    # torch.Size([2, 16, 3]) torch.Size([2, 256, 3])

    # model = StructureSetAbstractionMsg(npoint=16, radius_list=[0.2, 0.4, 0.8], nsample_list=[32, 64, 128],
    #                                    in_channel=3,
    #                                    mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
    #
    # out_1 = model(data, data_1,data)
    # plot_Matrix(out_1[0].detach().numpy())
    # print(out_1)
    # print(out.shape, out_1.shape)
    # torch.Size([2, 16, 3]) torch.Size([2, 256, 3])
