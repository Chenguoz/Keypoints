import importlib
import sys
import torch
import os
import argparse
import utils.check_points_utils as checkpoint_util
import numpy as np
from data_utils.dataset import Dataset
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


def capital_to_lower(dict_info):
    new_dict = {}
    for i, j in dict_info.items():
        new_dict[i.lower()] = j
    return new_dict


def caculate_miou(pre_label, seg_label, cat='Chair'):
    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                   'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                   'Mug': [36, 37],
                   'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
                   'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
    seg_classes = capital_to_lower(seg_classes)
    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]

    for l in seg_classes[cat]:
        if (np.sum(seg_label == l) == 0) and (
                np.sum(pre_label == l) == 0):  # part is not present, no prediction as well
            part_ious[l - seg_classes[cat][0]] = 1.0
        else:
            part_ious[l - seg_classes[cat][0]] = np.sum((seg_label == l) & (pre_label == l)) / float(
                np.sum((seg_label == l) | (pre_label == l)))
    # print(np.mean(part_ious))
    return np.mean(part_ious)


def main(args):
    if not os.path.exists('temp/'):
        os.mkdir('temp/')
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

    # if os.path.exists(args.output_dir) is False:
    #     os.makedirs(args.output_dir)

    test_dataset = Dataset(root=args.data_path, dataset_name=args.dataset_name, class_choice=category,
                           num_points=args.num_inputs, split='val',
                           segmentation=args.segmentation)

    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=36, shuffle=False,
                                                 num_workers=2, pin_memory=True,
                                                 persistent_workers=True)
    miou_list = []
    label_num = 1
    model_idx = args.model_idx

    point_cloud_jitter = PointcloudJitter(std=args.gauss_rate, clip=0.1)
    for batch_id, (batch_points, label, seg, name, file) in enumerate(testDataLoader, 0):
        with torch.no_grad():

            if not args.gauss_rate == 0:
                for pc in range(batch_points.shape[0]):
                    batch_points[pc] = point_cloud_jitter(batch_points[pc])
            structure_points, fps_points, cos_similarity, stpts_prob_map = model(batch_points.cuda())

            if label_num > 0:
                structure_points_i = structure_points[model_idx].cpu().detach().numpy()
                batch_points_i = batch_points[model_idx].cpu().detach().numpy()
                seg_i = seg[model_idx].cpu().detach().numpy()
                structure_points_index = knn_point(1, torch.from_numpy(batch_points_i[None, :, :]),
                                                   torch.from_numpy(structure_points_i[None, :, :]))[0]
                structure_points_label = seg_i[structure_points_index].squeeze()
                label_num -= 1


            for i in range(structure_points.shape[0]):
                structure_points_i = structure_points[i].cpu().detach().numpy()
                batch_points_i = batch_points[i].cpu().detach().numpy()
                fps_points_i = fps_points[i].cpu().detach().numpy()
                seg_i = seg[i].cpu().detach().numpy()

                points_index = knn_point(1, torch.from_numpy(structure_points_i[None, :, :]),
                                         torch.from_numpy(batch_points_i[None, :, :]))[0].squeeze()

                points_label = structure_points_label[points_index]

                miou_list.append(caculate_miou(points_label, seg_i, cat=category))

    print('{} avg miou: '.format(category + str(num_structure_points) + ' input' + str(args.num_inputs) + ' ' + str(args.model_idx)), np.mean(miou_list))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("-num_inputs", type=int, default=1024, help="sample points from initial point cloud")
    parser.add_argument("-log_dir", type=str, default='4.8.0', help="path to the trained model log")
    parser.add_argument("-model", type=str, default='model_min_test_loss',
                        help="the trained model[default: model_min_test_loss]")
    parser.add_argument("-output_dir", type=str, default='out', help="output dir")
    parser.add_argument("-test_on_aligned", type=str, default='True',
                        help="whether the testing shape is aligned or not. If set to False, the network should be trained with num_of_transform > 0 to use PCA data aug")
    parser.add_argument("-dataset_name", type=str, default='shapenetpart', help="")
    parser.add_argument("-data_path", type=str, default='../', help="")
    parser.add_argument('-segmentation', action='store_true', default=True, help='')
    parser.add_argument('-model_idx', type=int, default=0, help='')
    parser.add_argument('-gauss_rate', type=float, default=0, help='')

    args = parser.parse_args()
    main(args)
