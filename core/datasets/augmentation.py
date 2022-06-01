import numpy as np
import torch

def augment(pcd_data, grid_coords, config):
    '''
    rigid transformation
    '''

    if config['do_aug']:
        # rotation
        rotate_angle = config["rotate_angle"] / 180.0 * (np.pi)
        
        # for outdoor scene
        z_angle = np.random.uniform() * rotate_angle
        angles_2d = [0, 0, z_angle]

        # for indoor scene and object data
        angles_3d = np.random.rand(3) * rotate_angle

        N = pcd_data.shape[0]
        
        # shift
        shift_x = np.random.uniform(config['shift_x'][0], config['shift_x'][1], (1, 1))
        shift_y = np.random.uniform(config['shift_y'][0], config['shift_y'][1], (1, 1))
        shift_z = np.random.uniform(config['shift_z'][0], config['shift_z'][1], (1, 1))

        if grid_coords is not None:
            pcd_and_grid_data = np.concatenate((pcd_data, grid_coords), 0)
        else:
            pcd_and_grid_data = pcd_data
        
        # rotation 
        if config["rotate_dim"] == "2D":
            pcd_and_grid_data = atomic_rotate(pcd_and_grid_data, angles_2d)
        if config["rotate_dim"] == "3D":
            pcd_and_grid_data = atomic_rotate(pcd_and_grid_data, angles_3d)

        # shift
        if config["translation_perturbation"]:
            pcd_and_grid_data += np.concatenate([shift_x, shift_y, shift_z], 1)
    
    if grid_coords is not None:
        return pcd_and_grid_data[:N, :], pcd_and_grid_data[N:, :]
    else:
        return pcd_and_grid_data[:N, :], None


def angles2rotation_matrix(angles):
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
    return R


def atomic_rotate_pytorch_batch(data, angles):
    '''

    :param data:  Bx3xN tensor
    :param angles: list of [theta_x, theta_y, theta_z]
    :return:
    '''
    device = data.device
    B, N = data.size()[0], data.size()[2]

    R = torch.from_numpy(angles2rotation_matrix(angles).astype(np.float32)).to(device)  # 3x3
    R = R.unsqueeze(0).expand(B, 3, 3)

    rotated_data = torch.matmul(R, data)  # Bx3x3 * Bx3xN -> Bx3xN
    return rotated_data


def atomic_rotate_pytorch(data, angles):
    '''

    :param data: 3xN tensor
    :param angles: list of [theta_x, theta_y, theta_z]
    :return:
    '''
    device = data.device
    N = data.size()[1]

    R = torch.from_numpy(angles2rotation_matrix(angles).astype(np.float32)).to(device)  # 3x3

    rotated_data = torch.matmul(R, data)  # 3x3 * 3xN -> 3xN
    return rotated_data


def atomic_rotate(data, angles):
    '''

    :param data: numpy array of Nx3 array
    :param angles: numpy array / list of 3
    :return: rotated_data: numpy array of Nx3
    '''
    R = angles2rotation_matrix(angles)
    rotated_data = np.dot(data, R)

    return rotated_data


def rotate_point_cloud_90(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    y_angle = np.random.randint(low=0, high=4) * (np.pi/2.0)
    angles = [0, y_angle, 0]
    rotated_data = atomic_rotate(data, angles)

    return rotated_data


def rotate_point_cloud_up(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    y_angle = np.random.uniform() * 2 * np.pi
    angles = [0, y_angle, 0]
    rotated_data = atomic_rotate(data, angles)

    return rotated_data


def rotate_point_cloud_3d(data):
    # uniform sampling
    angles = np.random.rand(3) * np.pi * 2
    rotated_data = atomic_rotate(data, angles)

    return rotated_data


def rotate_point_cloud_list_3d(pc_list, angles=None):
    if angles is None:
        # uniform sampling
        angles = np.random.rand(3) * np.pi * 2

    rotated_pc_list = []
    for pc in pc_list:
        rotated_pc_list.append(atomic_rotate(pc, angles))
    return rotated_pc_list


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, jittered point clouds
    """
    N, C = data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += data
    return jittered_data


def transform_pc_pytorch(pc, sn, node, rot_type='2d', scale_thre=0.2, shift_thre=0.2, rot_perturbation=False):
    '''

    :param pc: 3xN tensor
    :param sn: 5xN tensor / 4xN tensor
    :param node: 3xM tensor
    :return: pc, sn, node of the same shape, detach
    '''
    device = pc.device
    # N, M = pc.size()[1], node.size()[1]

    # 1. rotate around the up axis
    if rot_type == '2d':
        x_angle, z_angle = 0, 0
        y_angle = np.random.uniform() * 2 * np.pi
    elif rot_type == '3d':
        x_angle = np.random.uniform() * 2 * np.pi
        y_angle = np.random.uniform() * 2 * np.pi
        z_angle = np.random.uniform() * 2 * np.pi
    elif rot_type is None:
        x_angle, y_angle, z_angle = 0, 0, 0
    else:
        raise Exception('Invalid rot_type.')

    if rot_perturbation == True:
        angle_sigma = 0.06
        angle_clip = 3 * angle_sigma
        x_angle += np.clip(angle_sigma * np.random.randn(), -angle_clip, angle_clip)
        y_angle += np.clip(angle_sigma * np.random.randn(), -angle_clip, angle_clip)
        z_angle += np.clip(angle_sigma * np.random.randn(), -angle_clip, angle_clip)

    angles = [x_angle, y_angle, z_angle]
    R = torch.from_numpy(angles2rotation_matrix(angles).astype(np.float32)).to(device)  # 3x3
    pc = torch.matmul(R, pc)  # 3x3 * 3xN -> 3xN
    if sn.size()[0] >= 3:
        sn[0:3, :] = torch.matmul(R, sn[0:3, :])  # 3x3 * 3xN -> 3xN
    node = torch.matmul(R, node)  # 3x3 * 3xN -> 3xN

    # 2. scale
    scale = np.random.uniform(low=1-scale_thre, high=1+scale_thre)
    pc = pc * scale
    node = node * scale

    # 3. translation
    shift = torch.from_numpy(np.random.uniform(-1*shift_thre, shift_thre, (3, 1)).astype(np.float32)).to(device)  # 3x1
    pc = pc + shift
    node = node + shift

    return pc.detach(), sn.detach(), node.detach(), \
           R, scale, shift

 


