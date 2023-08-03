import tqdm
import importlib
import sys
import torch
import os
import argparse
import utils.check_points_utils as checkpoint_util
import numpy as np
from tqdm import tqdm
from data_utils.keypointnet_dataloader import KeyPointNetDataLoader
from models.torch_pointnet_utils import knn_point


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


def main(args):
    experiment_dir = 'log/' + args.log_dir
    sys.path.append(experiment_dir)
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]

    epoch, iters, checkpoint = checkpoint_util.load_checkpoint(model_3d=None, filename=str(
        experiment_dir) + '/checkpoints/' + args.model)
    category = checkpoint['category']
    num_structure_points = checkpoint['num_structure_points']
    multi_distribution = checkpoint['multi_distribution']
    offset = checkpoint['offset']

    model = importlib.import_module(model_name)
    model = model.Pointnet2StructurePointNet(num_structure_points=num_structure_points, input_channels=0,
                                             multi_distribution_num=multi_distribution,
                                             offset=offset)
    model.load_state_dict(checkpoint['model_state_3d'])

    model.cuda()
    model.eval()

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    test_dataset = KeyPointNetDataLoader(num_points=args.num_inputs, json_path=os.path.join(args.json_path, category + '.json'),
                                         pcd_path=args.pcd_path, split='val')

    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False,
                                                 num_workers=2, pin_memory=True,
                                                 persistent_workers=True)

    thresholds = np.linspace(0., 1)

    model_datas = []
    point_cloud_jitter = PointcloudJitter(std=args.gauss_rate, clip=0.1)

    for batch_id, (batch_points, key_points, key_points_num, _) in tqdm(enumerate(testDataLoader, 0),
                                                                        total=len(testDataLoader),
                                                                        smoothing=0.9):

        with torch.no_grad():
            batch_points_jitter = torch.Tensor(batch_points)
            if not args.gauss_rate == 0:
                for pc in range(batch_points.shape[0]):
                    batch_points_jitter[pc] = point_cloud_jitter(batch_points[pc])

            structure_points, fps_points, cos_similarity, stpts_prob_map = model(batch_points_jitter.cuda())
            for i in range(0, batch_points.shape[0]):
                diameter_shape = torch.sqrt(torch.sum((torch.max(batch_points[i], dim=0)[0] - torch.min(batch_points[i], dim=0)[0]) ** 2))

                model_datas.append({'structure_pts': structure_points[i] + (torch.rand(structure_points[i].shape) * args.noise_rate).cuda(),
                                    'gt_feat_pts': key_points[i][:key_points_num[i], :].cuda(), 'diameter_shape': diameter_shape})

    dis_ratios, dis_thresholds = compute_correspondence_accuracy(model_datas)

    if not os.path.exists('corrs_model/'):
        os.mkdir('corrs_model/')
    np.savez('corrs_model/' + args.corres_name, dis_ratios, dis_thresholds)


def compute_correspondence_dis(model_data_a, model_data_b):
    structure_pts_a = model_data_a['structure_pts']
    gt_feat_pts_a = model_data_a['gt_feat_pts']
    structure_pts_b = model_data_b['structure_pts']
    gt_feat_pts_b = model_data_b['gt_feat_pts']
    diameter_shape_b = model_data_b['diameter_shape']
    res_dis = []

    if not gt_feat_pts_a.shape[0] == gt_feat_pts_b.shape[0]:
        return res_dis

    # knn_a_idxs, knn_a_dis = query_KNN_tensor(structure_pts_a, gt_feat_pts_a, 1)
    knn_a_idxs = knn_point(1, structure_pts_a[None, :, :], gt_feat_pts_a[None, :, :])

    corres_pts_in_b = structure_pts_b[knn_a_idxs[0, :, 0], :]
    diff = corres_pts_in_b - gt_feat_pts_b
    tmp_dis = torch.sqrt(torch.sum(diff * diff, dim=1)) / diameter_shape_b

    for i in range(tmp_dis.shape[0]):
        # nan means this feature point is missing on groundtruth model
        if torch.isnan(gt_feat_pts_a[i, 0]) == False and torch.isnan(gt_feat_pts_b[i, 0]) == False:
            res_dis.append(tmp_dis[i].item())

    return res_dis


def compute_correspondence_accuracy(model_datas):
    dis_list = []
    for i in tqdm(range(len(model_datas)), total=len(model_datas)):
        for j in range(len(model_datas)):
            if i == j:
                continue
            model_data_i = model_datas[i]
            model_data_j = model_datas[j]
            corres_dis = compute_correspondence_dis(model_data_i, model_data_j)
            dis_list = dis_list + corres_dis

    dis_array = np.array(dis_list)

    dis_thresholds = np.arange(0, 0.26, 0.01)
    dis_ratios = []

    for i in range(dis_thresholds.shape[0]):
        threshold = dis_thresholds[i]
        ratio = dis_array[dis_array <= threshold].shape[0] / dis_array.shape[0]
        dis_ratios.append(ratio)

    dis_ratios = np.array(dis_ratios)

    return dis_ratios, dis_thresholds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("-num_inputs", type=int, default=1024, help="sample points from initial point cloud")
    parser.add_argument("-log_dir", type=str, default='4.8.2', help="path to the trained model log")
    parser.add_argument("-model", type=str, default='model_min_test_loss',
                        help="the trained model[default: model_min_test_loss]")
    parser.add_argument("-output_dir", type=str, default='out', help="output dir")
    parser.add_argument('-prediction_output', type=str, default='merger_prediction.npz',
                        help='Output file where prediction results are written.')
    parser.add_argument('-pcd_path', type=str, default='./keypointnet/pcds',
                        help='Point cloud file folder path from KeypointNet dataset.')
    parser.add_argument('-json_path', default='./keypointnet/annotations/', help='')
    parser.add_argument('-gauss_rate', type=float, default=0, help='')
    parser.add_argument('-noise_rate', type=float, default=0, help='')
    parser.add_argument('-corres_name', type=str, help='')

    args = parser.parse_args()
    main(args)
