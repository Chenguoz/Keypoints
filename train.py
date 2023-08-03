import sys
import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import gc
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

import utils.check_points_utils as checkpoint_util
from tqdm import tqdm
import logging
import datetime
from pathlib import Path
import provider
from data_utils.dataset import Dataset
from data_utils.keypointnet_dataloader import KeyPointNetDataLoader

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast


def train_one_epoch(model, optimizer, data_loader, current_iter, criterions, lr_scheduler, num_of_trans,
                    num_inputs, logger):
    model.train()
    loss_dict = {}
    loss_dict['Loss'] = 0
    for l in criterions.keys():
        loss_dict[l] = 0
    count = 0

    for batch_id, (batch_points, _, _, _) in tqdm(enumerate(data_loader, 0), total=len(data_loader),
                                                  smoothing=0.9):
        optimizer.zero_grad()
        # print(batch_points.shape)
        batch_points = batch_points.data.numpy()
        batch_points[:, :, 0:3] = provider.random_scale_point_cloud(batch_points[:, :, 0:3])
        batch_points[:, :, 0:3] = provider.shift_point_cloud(batch_points[:, :, 0:3])
        batch_points = torch.Tensor(batch_points)
        batch_points = batch_points.cuda()

        if args.use_half:
            with autocast():
                structure_points, fps_points, cos_similarity, stpts_prob_map = model(batch_points)

                ComputeLoss3dLoss = criterions['ComputeLoss3d'](batch_points, structure_points)
                WeightedChamferLoss = criterions['WeightedChamferLoss'](fps_points, structure_points, stpts_prob_map, batch_points)
                # loss_Vec = criterions['VecLoss'](structure_points, cos_similarity, 0.85)
                loss = ComputeLoss3dLoss+WeightedChamferLoss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            structure_points, fps_points, cos_similarity, stpts_prob_map = model(batch_points)

            ComputeLoss3dLoss = criterions['ComputeLoss3d'](batch_points, structure_points)
            WeightedChamferLoss = criterions['WeightedChamferLoss'](fps_points, structure_points, stpts_prob_map, batch_points)
            # loss_Vec = criterions['VecLoss'](structure_points, cos_similarity, 0.85)
            loss = ComputeLoss3dLoss+WeightedChamferLoss

            loss.backward()
            optimizer.step()

        current_iter += 1
        loss_dict['Loss'] += loss.item()
        loss_dict['ComputeLoss3d'] += ComputeLoss3dLoss.item()
        loss_dict['WeightedChamferLoss'] += WeightedChamferLoss.item()
        # loss_dict['VecLoss'] += loss_Vec.item()

        current_iter += 1
        # gc.collect()
        count += 1

    lr_scheduler.step()
    for k in loss_dict.keys():
        loss_dict[k] /= count

    return loss_dict, current_iter


def test(model, data_loader, criterions):
    model.eval()
    count = 0
    loss_dict = {}
    loss_dict['Loss'] = 0
    for l in criterions.keys():
        loss_dict[l] = 0
    for batch_id, (batch_points, _, _, _) in tqdm(enumerate(data_loader, 0), total=len(data_loader),
                                                  smoothing=0.9):
        batch_points = batch_points.cuda()
        structure_points, fps_points, cos_similarity, stpts_prob_map = model(batch_points)

        ComputeLoss3dLoss = criterions['ComputeLoss3d'](batch_points, structure_points)
        WeightedChamferLoss = criterions['WeightedChamferLoss'](fps_points, structure_points, stpts_prob_map, batch_points)
        # loss_Vec = criterions['VecLoss'](structure_points, cos_similarity, 0.85)
        loss = WeightedChamferLoss

        loss_dict['Loss'] += loss.item()
        loss_dict['ComputeLoss3d'] += ComputeLoss3dLoss.item()
        loss_dict['WeightedChamferLoss'] += WeightedChamferLoss.item()
        # loss_dict['VecLoss'] += loss_Vec.item()

        count += 1

    for k in loss_dict.keys():
        loss_dict[k] /= count
    return loss_dict


def create_loggger(args):
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    # exp_dir = exp_dir.joinpath('')
    # exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model), mode='w+')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, checkpoints_dir, exp_dir


def log_string(str, logger):
    logger.info(str)
    print(str)


def train(args):
    # lr_clip = 1e-5
    # bnm_clip = 1e-2
    torch.cuda.empty_cache()
    logger, checkpoints_dir, exp_dir = create_loggger(args)
    log_string('PARAMETER ...', logger=logger)
    log_string(args, logger=logger)

    '''DATA LOADING'''
    log_string('Load dataset ...', logger=logger)

    # train_dataset = bhcp_dataloader(args.data_path, args.category, is_pts_aligned=False, split='train')
    # test_dataset = bhcp_dataloader(args.data_path, args.category, is_pts_aligned=False, split='test')
    # train_dataset = KeyPointNetDataLoader(json_path=cmd_args.json_path, pcd_path=cmd_args.pcd_path, split='train')
    # test_dataset = KeyPointNetDataLoader(json_path=cmd_args.json_path, pcd_path=cmd_args.pcd_path, split='val')

    if args.dataset_name == 'keypointnet':
        train_dataset = KeyPointNetDataLoader(num_points=args.num_inputs, json_path=os.path.join(args.json_path, args.category + '.json'),
                                              pcd_path=args.pcd_path, split='train')
        test_dataset = KeyPointNetDataLoader(num_points=args.num_inputs, json_path=os.path.join(args.json_path, args.category + '.json'),
                                             pcd_path=args.pcd_path, split='val')
    else:
        train_dataset = Dataset(root=args.data_path, dataset_name=args.dataset_name, class_choice=args.category,
                                num_points=args.num_inputs, split='train',
                                segmentation=args.segmentation)
        test_dataset = Dataset(root=args.data_path, dataset_name=args.dataset_name, class_choice=args.category,
                               num_points=args.num_inputs, split='test',
                               segmentation=args.segmentation)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, drop_last=True, pin_memory=True,
                                                  persistent_workers=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('./models/chamfer_distance.py', str(exp_dir))
    shutil.copy('./models/torch_pointnet_utils.py', str(exp_dir))
    shutil.copy('./train.py', str(exp_dir))

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    criterions = {'ComputeLoss3d': model.ComputeLoss3d(), 'WeightedChamferLoss': model.WeightedChamferLoss(),
                  'VecLoss': model.VecLoss()}
    # criterions = {'ComputeLoss3d': model.ComputeLoss3d(), 'VecLoss': model.VecLoss()}

    model = model.Pointnet2StructurePointNet(num_structure_points=args.num_structure_points, input_channels=0,
                                             multi_distribution_num=args.multi_distribution,
                                             offset=args.offset)

    model.cuda()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_batch, gamma=args.lr_decay)

    iters = -1
    min_test_loss = float('inf')

    # load status from checkpoint
    start_epoch = 0
    if args.checkpoint is not None:
        start_epoch, iters = checkpoint_util.load_checkpoint(model_3d=model, optimizer=optimizer,
                                                             filename=args.checkpoint)
        start_epoch += 1

    log_string('Start Training Unsupervised Structure Points for %s...' % args.dataset_name, logger=logger)
    iters = max(iters, 0)

    for epoch_i in range(start_epoch, args.max_epochs):
        log_string('-------------------------------------------', logger=logger)
        log_string('Epoch %d/%s,Learning Rate %f:' % (epoch_i + 1, args.max_epochs, lr_scheduler.get_last_lr()[0]),
                   logger=logger)

        loss_dict, iters = train_one_epoch(model,
                                           optimizer,
                                           trainDataLoader,
                                           iters,
                                           criterions,
                                           lr_scheduler,
                                           num_of_trans=args.num_of_transform,
                                           num_inputs=args.num_inputs,
                                           logger=logger)
        loss_str = ''
        for i in loss_dict.keys():
            loss_str += '%s: %f \t' % (i, loss_dict[i])
        log_string(loss_str, logger)

        with torch.no_grad():
            loss_dict = test(model,
                             data_loader=testDataLoader,
                             criterions=criterions,
                             )
            loss_str = ''
            for i in loss_dict.keys():
                loss_str += '%s: %f \t' % (i, loss_dict[i])
            log_string(loss_str, logger)

            if loss_dict['Loss'] < min_test_loss:
                min_test_loss = loss_dict['Loss']
                log_string('Min Test Loss: %f' % (loss_dict['Loss']), logger=logger)

                log_string('Save model...', logger=logger)
                fname = os.path.join(checkpoints_dir, 'model_min_test_loss')
                checkpoint_util.save_checkpoint(filename=fname, model_3d=model, optimizer=optimizer, iters=iters,
                                                epoch=epoch_i, category=args.category,
                                                num_structure_points=args.num_structure_points,
                                                multi_distribution=args.multi_distribution,
                                                offset=args.offset)
            else:
                log_string('Min Test Loss: %f' % (min_test_loss), logger=logger)

        if (epoch_i + 1) % 50 == 0:
            fname = os.path.join(checkpoints_dir, 'model_%d' % (epoch_i + 1))
            checkpoint_util.save_checkpoint(filename=fname, model_3d=model, optimizer=optimizer, iters=iters,
                                            epoch=epoch_i, category=args.category,
                                            num_structure_points=args.num_structure_points,
                                            multi_distribution=args.multi_distribution,
                                            offset=args.offset)
        fname = os.path.join(checkpoints_dir, 'model')
        checkpoint_util.save_checkpoint(filename=fname, model_3d=model, optimizer=optimizer, iters=iters,
                                        epoch=epoch_i, category=args.category,
                                        num_structure_points=args.num_structure_points,
                                        multi_distribution=args.multi_distribution,
                                        offset=args.offset)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("-batch_size", type=int, default=36, help="Batch size")
    parser.add_argument("-weight_decay", type=float, default=1e-5, help="L2 regularization coeff")
    parser.add_argument("-num_inputs", type=int, default=1024, help="sample points from initial point cloud")
    parser.add_argument("-num_structure_points", type=int, default=12, help="Number of structure points")
    parser.add_argument("-category", type=str, default='laptop', help="Category of the objects to train")
    parser.add_argument("-dataset_name", type=str, default='shapenetpart', help="keypointnet,shapenetpart")
    parser.add_argument("-data_path", type=str, default='../', help="")
    parser.add_argument('-segmentation', action='store_true', default=False, help='')
    parser.add_argument('-offset', action='store_true', default=False, help='')
    parser.add_argument("-max_epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("-log_dir", type=str, default=None, help="Root of the log")
    parser.add_argument("-multi_distribution", type=int, default=3, help="Multivariate normal distribution nums")
    parser.add_argument('-num_workers', type=int, default=4, help='dataload num worker')
    parser.add_argument('-model', default='model_weightchamfer', help='model name [default: model_weightchamfer Structure_pointnet]')
    parser.add_argument('-use_half', action='store_true', default=True, help='use mix half mode')
    parser.add_argument('-json_path', default='./keypointnet/annotations/', help='')
    parser.add_argument('-pcd_path', type=str, default='./keypointnet/pcds',
                        help='Point cloud file folder path from KeypointNet dataset.')
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("-lr_decay", type=float, default=0.7, help="Learning rate decay gamma")
    parser.add_argument("-decay_batch", type=float, default=20, help="Learning rate decay batch")
    parser.add_argument("-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum")
    parser.add_argument("-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma")
    parser.add_argument("-checkpoint_save_step", type=int, default=50, help="Step for saving Checkpoint")
    parser.add_argument("-checkpoint", type=str, default=None, help="Checkpoint to start from")
    parser.add_argument("-num_of_transform", type=int, default=0,
                        help="Number of transforms for rotation data augmentation. Useful when testing on shapes without alignment")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    import platform

    sys = platform.system()
    if sys == "Windows":
        args.batch_size = 2

    train(args=args)
