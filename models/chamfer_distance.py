from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
from torch_pointnet_utils import query_ball_point, index_points
import torch.nn.functional as F


def query_KNN_tensor(points, query_pts, k):
    '''

       :param points: bn x n x 3
       :param query_pts: bn x m x 3
       :param k: num of neighbors
       :return: nb x m x k  ids, sorted_squared_dis
       '''

    diff = query_pts[:, :, None, :] - points[:, None, :, :]

    squared_dis = torch.sum(diff * diff, dim=3)  # bn x m x n
    sorted_squared_dis, sorted_idxs = torch.sort(squared_dis, dim=2)
    sorted_idxs = sorted_idxs[:, :, :k]
    sorted_squared_dis = sorted_squared_dis[:, :, :k]

    return sorted_idxs, sorted_squared_dis


def compute_chamfer_distance(p1, p2):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[bn, N, D]
    :param p2: size[bn, M, D]
    :return: sum of Chamfer Distance of two point sets
    '''

    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = torch.sum(diff * diff, dim=3)
    dist1 = dist
    dist2 = torch.transpose(dist, 1, 2)

    dist_min1, _ = torch.min(dist1, dim=2)
    dist_min2, _ = torch.min(dist2, dim=2)

    return dist_min1, dist_min2


# def compute_vector_distance(p1, p2):
#     '''
#     Calculate Chamfer Distance between two point sets
#     :param p1: size[bn, N, D]
#
#     :return: sum of Chamfer Distance of two point sets
#     '''
#     # device = p1.device
#
#     vectors = p1[:, :, None, :] - p1[:, None, :, :]
#     # vectors=vectors.float()
#     vectors = torch.flatten(vectors, start_dim=1, end_dim=2)
#     vectors_dot = torch.sum(vectors[:, :, None, :] * vectors[:, None, :, :], dim=3)
#     vectors_dot = torch.abs(vectors_dot)
#
#
#     # edge_dict = compute_edge_distance(p1, p2, maxdis=0.001)
#     # idx = torch.zeros(vectors_dot.shape).cuda()
#     # idx_num = 0
#     # for key, values in edge_dict.items():
#     #     for value in values:
#     #         idx[key, value[0], value[1]] = 1
#     #         idx_num += 1
#     #
#     # vectors_dot = vectors_dot * idx
#
#
#     # print(vectors_dot[vectors_dot>0])
#     vectors_abs = torch.sqrt(torch.sum(vectors * vectors, dim=2, keepdim=True))
#     vectors_abs_dot = torch.sum(vectors_abs[:, :, None, :] * vectors_abs[:, None, :, :], dim=3)
#     vectors_abs_dot[vectors_abs_dot == 0] = 1
#     vectors_cos = vectors_dot / vectors_abs_dot
#
#     return torch.mean(vectors_cos)
#
#     # return torch.sum(vectors_cos) / idx_num


def compute_vector_distance(p1, p2=None):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[bn, N, D]
    :param p2: size[bn, M, D]
    :return: sum of Chamfer Distance of two point sets
    '''

    vectors = p1[:, :, None, :] - p1[:, None, :, :]

    vectors = torch.flatten(vectors, start_dim=1, end_dim=2)
    vectors_dot = torch.sum(vectors[:, :, None, :] * vectors[:, None, :, :], dim=3)
    vectors_dot = torch.abs(vectors_dot)

    edge_dict = compute_edge_distance(p1, p2, maxdis=0.001)
    idx = torch.zeros(vectors_dot.shape).cuda()
    idx_num = 1
    for key, values in edge_dict.items():
        for value in values:
            idx[key, value[0], value[1]] = 1
            idx_num += 1
    vectors_abs = torch.sqrt(torch.sum(vectors * vectors + 1e-5, dim=2, keepdim=True))
    vectors_dot = vectors_dot * idx
    # vectors_abs = torch.sum(vectors * vectors, dim=2, keepdim=True)
    vectors_abs_dot = torch.sum(vectors_abs[:, :, None, :] * vectors_abs[:, None, :, :], dim=3)
    vectors_abs_dot[vectors_abs_dot == 0] = 1
    vectors_cos = vectors_dot / vectors_abs_dot
    return torch.sum(vectors_cos) / idx_num


def compute_edge_distance(p1, p2, maxdis=0.001):
    point_num = p1.shape[1]
    edge_list = []
    batch_dict = {}
    # for batch in range(p1.shape[0]):
    for i in range(point_num):
        for j in range(i):
            start = p1[:, i, :]
            end = p1[:, j, :]
            dist = torch.mean(torch.sqrt(1e-3 + (torch.sum(torch.square(start - end), dim=-1))))
            count = 5
            device = dist.device
            f_interp = torch.linspace(0.0, 1.0, count).unsqueeze(0).unsqueeze(-1).to(device)
            b_interp = 1.0 - f_interp
            K = start.unsqueeze(-2) * f_interp + end.unsqueeze(-2) * b_interp
            dist1, dist2 = compute_chamfer_distance(K, p2)
            cdis = (torch.mean(dist1, dim=1))
            for x in torch.where(cdis < maxdis)[0].cpu().numpy():
                edge_list.append([x, i, j])

    for x in edge_list:
        if x[0] in batch_dict:
            batch_dict[x[0]].append(x[1:])
        else:
            batch_dict[x[0]] = [x[1:]]

    return batch_dict


def compute_offset_distance(p1):
    return torch.sum(p1 * p1)


def compute_vector_similarity_distance(p1, similarity_map, threshold):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[bn, N, D]
    :param p2: size[bn, M, D]
    :return: sum of Chamfer Distance of two point sets
    '''
    # B, N, D = p1.shape
    # device = p1.device
    #
    # vectors = p1[:, :, None, :] - p1[:, None, :, :]
    # similarity_map[similarity_map < threshold] = 0
    # vectors = vectors * similarity_map.view(B, N, N, 1).to(device)
    # vectors = torch.flatten(vectors, start_dim=1, end_dim=2)
    #
    # vectors_abs = torch.sum(vectors * vectors, dim=2, keepdim=True)
    # edge_idx = torch.nonzero(vectors_abs)
    # loss = 0
    # edge_num = 0
    # for b in range(B):
    #     b_edge = edge_idx[edge_idx[:, 0] == b]
    #     edges = torch.empty((b_edge.shape[0], D))
    #     edge_num += b_edge.shape[0] * b_edge.shape[0]
    #     for i, edge in enumerate(b_edge):
    #         edges[i, :] = vectors[b, edge[1]]
    #     for i in range(b_edge.shape[0]):
    #         loss += torch.sum(torch.abs(F.cosine_similarity(edges[i, :].view(1, -1), edges, dim=1)))

    B, N, D = p1.shape
    device = p1.device
    vectors = p1[:, :, None, :] - p1[:, None, :, :]
    similarity_map[similarity_map < threshold] = 0
    vectors = vectors * similarity_map.view(B, N, N, 1).to(device)
    # vectors[similarity_map < threshold] = 0
    vectors = torch.flatten(vectors, start_dim=1, end_dim=2)
    vectors_dot = torch.sum(vectors[:, :, None, :] * vectors[:, None, :, :], dim=3)
    idx_num = torch.sum(vectors_dot != 0)
    vectors_dot = torch.abs(vectors_dot)
    vectors_abs = torch.sqrt(torch.sum(vectors * vectors + 1e-9, dim=2, keepdim=True))
    vectors_abs_dot = torch.sum(vectors_abs[:, :, None, :] * vectors_abs[:, None, :, :], dim=3)
    vectors_abs_dot[vectors_abs_dot == 0] = 1
    vectors_cos = vectors_dot / vectors_abs_dot
    return torch.sum(vectors_cos) / idx_num


def compute_end_distance(p1, p2, radius=0.1, nsample=16):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[bn, N, D]
    :param p2: size[bn, M, D]
    :return: sum of Chamfer Distance of two point sets
    '''
    B, S, C = p1.shape
    idx = query_ball_point(radius, nsample, p2, p1)
    p2 = index_points(p2, idx)  # [B, npoint, nsample, C]
    p2_norm = p2 - p1.view(B, S, 1, C)
    # p2_norm = p2_norm / torch.norm(p2_norm, p=2, dim=3, keepdim=True)
    dist = torch.sum(p2_norm, dim=2)
    dist = torch.sqrt(torch.sum(dist * dist, dim=2))
    return torch.mean(dist)


class ComputeCDLoss(nn.Module):
    def __init__(self):
        super(ComputeCDLoss, self).__init__()

    def forward(self, recon_points, gt_points):
        dist1, dist2 = compute_chamfer_distance(recon_points, gt_points)
        loss = (torch.sum(dist1) + torch.sum(dist2)) / ((recon_points.shape[0]) * gt_points.shape[1]) * 1024
        # print(torch.sum(dist1), torch.sum(dist2))
        # loss = (torch.mean(dist1) + torch.mean(dist2) * 32)

        return loss


class ComputeVecLoss(nn.Module):
    def __init__(self):
        super(ComputeVecLoss, self).__init__()

    def forward(self, recon_points, gt_points):
        dist3 = compute_vector_distance(recon_points, gt_points)
        # loss_align = torch.sum(dist3) / (recon_points.shape[0])
        return dist3


class ComputeEdgeLoss(nn.Module):
    def __init__(self):
        super(ComputeEdgeLoss, self).__init__()

    def forward(self, recon_points, gt_points):
        dist3 = compute_edge_distance(recon_points, gt_points)
        # loss_align = torch.sum(dist3) / (recon_points.shape[0])
        return dist3


class ComputeOffsetLoss(nn.Module):
    def __init__(self):
        super(ComputeOffsetLoss, self).__init__()

    def forward(self, fps_points_offset):
        return compute_offset_distance(fps_points_offset) / (fps_points_offset.shape[0])


class ComputeVecSimilarityLoss(nn.Module):
    def __init__(self):
        super(ComputeVecSimilarityLoss, self).__init__()

    def forward(self, gt_points, cos_similarity, threshold):
        return compute_vector_similarity_distance(gt_points, cos_similarity, threshold)


class ComputeEndLoss(nn.Module):
    def __init__(self):
        super(ComputeEndLoss, self).__init__()

    def forward(self, recon_points, gt_points):
        dist1 = compute_end_distance(recon_points, gt_points, radius=0.1, nsample=16)
        loss = dist1 / recon_points.shape[1] * 24

        return loss


if __name__ == '__main__':
    p1 = torch.rand((2, 16, 3))
    p2 = torch.rand((2, 128, 3))
    p3 = torch.rand(2, 16, 16)
    a = ComputeVecSimilarityLoss()

    print(a(p1, p3, 0.95))
