import numpy as np
import matplotlib.pyplot as plt
import random


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


def paint(points_xyz, point_cloud, cos_similarity=None, show_similarity=False):
    # 24*6*1024
    # points_xyz[0, :3, :]
    # 3*1024
    # points.transpose((1, 0))
    # 1024*3

    x1 = points_xyz[:, 0]
    y1 = points_xyz[:, 1]
    z1 = points_xyz[:, 2]

    x2 = point_cloud[:, 0]
    y2 = point_cloud[:, 1]
    z2 = point_cloud[:, 2]

    if show_similarity:
        plot_Matrix(cos_similarity)

        for j in range(points_xyz.shape[0]):
            if j == 3:
                break
            ax1 = plt.subplot(111, projection='3d')
            ax1.axis('off')
            for i in np.where(cos_similarity[j] > 0.95):
                ax1.scatter(x1[i], y1[i], z1[i], s=200)
            ax1.scatter(x1, y1, z1, c=COLOR_LIST[:points_xyz.shape[0], :] / 255, s=48)
            ax1.scatter(x2, y2, z2, c='#A9A9A9', s=2)
            plt.show()
    else:
        ax1 = plt.subplot(111, projection='3d')
        ax1.scatter(x1, y1, z1, c=COLOR_LIST[:points_xyz.shape[0], :] / 255, s=48)
        ax1.scatter(x2, y2, z2, c='#A9A9A9', s=1)
        ax1.axis('off')
        plt.show()


def paint_map(points_xyz, point_cloud, stpts_prob_map):
    x1 = points_xyz[:, 0]
    y1 = points_xyz[:, 1]
    z1 = points_xyz[:, 2]

    x2 = point_cloud[:, 0]
    y2 = point_cloud[:, 1]
    z2 = point_cloud[:, 2]
    fig, ax = plt.subplots()

    point_map = np.sum(stpts_prob_map, axis=0)
    point_map = (point_map - np.min(point_map)) / (np.max(point_map) - np.min(point_map))
    ax1 = plt.subplot(111, projection='3d')
    # ax1.scatter(x1, y1, z1, c=COLOR_LIST[:points_xyz.shape[0], :] / 255, s=48)
    ax1.scatter(x1, y1, z1, c='gray', s=48)

    ax1.scatter(x2, y2, z2, c=plt.get_cmap('YlOrRd')(point_map), s=48)
    ax1.axis('off')
    plt.show()


def paint_seg(points_xyz, point_cloud, points_label, points_true_label):
    # 24*6*1024
    # points_xyz[0, :3, :]
    # 3*1024
    # points.transpose((1, 0))
    # 1024*3
    COLOR_LIST = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                           [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],
                           [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],
                           [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                           [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                           [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                           [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                           [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                           [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                           [1.00000000e+00, 0.00000000e+00, 9.37500000e-02]])*255
    from models.torch_pointnet_utils import knn_point, index_points
    x1 = points_xyz[:, 0]
    y1 = points_xyz[:, 1]
    z1 = points_xyz[:, 2]

    x2 = point_cloud[:, 0]
    y2 = point_cloud[:, 1]
    z2 = point_cloud[:, 2]

    # points_index = knn_point(1, torch.from_numpy(points_xyz[None, :, :]), torch.from_numpy(point_cloud[None, :, :]))[0].squeeze()
    # structure_points_index = knn_point(1, torch.from_numpy(point_cloud[None, :, :]), torch.from_numpy(points_xyz[None, :, :]))[0]
    # structure_points_label = seg_label[structure_points_index].squeeze()
    # points_label = structure_points_label[points_index]
    ax1 = plt.subplot(121, projection='3d')
    ax1.set_title('Predict')
    ax2 = plt.subplot(122, projection='3d')
    ax2.set_title('True')

    # ax1.scatter(x1, y1, z1, c=COLOR_LIST[:points_xyz.shape[0], :] / 255, s=48)
    # ax1.scatter(x1, y1, z1, c=COLOR_LIST[structure_points_label, :] / 255, s=48)
    points_label = points_label - np.min(points_label)
    points_true_label = points_true_label - np.min(points_true_label)

    # ax1.scatter(x2, y2, z2, c=COLOR_LIST[seg_label, :] / 255, s=4)
    ax1.scatter(x2, y2, z2, c=COLOR_LIST[points_label, :] / 255, s=4)
    ax2.scatter(x2, y2, z2, c=COLOR_LIST[points_true_label, :] / 255, s=4)
    # ax1.scatter(x2, y2, z2, c='#A9A9A9', s=1)
    ax1.axis('off')
    ax2.axis('off')

    plt.show()


def create_color_list(num):
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


COLOR_LIST = create_color_list(1024)


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k, :, 0:3]
        shape_normal = batch_xyz_normal[k, :, 3:6]
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc
