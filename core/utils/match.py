import numpy as np
import torch
import open3d as o3d
from torch_scatter import scatter_max
from core.nets.common import normalize_3d_coordinate, coordinate2index

import ipdb


def nms(keypoints_np, sals_np, NMS_radius, total_num=2000):
    '''

    :param keypoints_np: Mx3
    :param sigmas_np: M
    :return: valid_keypoints_np, valid_sigmas_np, valid_descriptors_np
    '''
    if NMS_radius < 0.01:
        if keypoints_np.shape[0] > total_num:
            sorted_idx = np.argsort(-sals_np)
            sorted_idx = sorted_idx[0:total_num]
            return keypoints_np[sorted_idx], sals_np[sorted_idx]  # Mx3
        else:
            return keypoints_np, sals_np

    valid_keypoint_counter = 0
    valid_keypoints_np = np.zeros(keypoints_np.shape, dtype=keypoints_np.dtype)
    valid_sals_np = np.zeros(sals_np.shape, dtype=sals_np.dtype)

    while keypoints_np.shape[0] > 0 and valid_keypoint_counter < total_num:
        # print(sigmas_np.shape)
        # print(sigmas_np)

        min_idx = np.argmax(sals_np, axis=0)
        # print(min_idx)

        valid_keypoints_np[valid_keypoint_counter, :] = keypoints_np[min_idx, :]
        valid_sals_np[valid_keypoint_counter] = sals_np[min_idx]
        # remove the rows that within a certain radius of the selected minimum
        distance_array = np.linalg.norm(
            (valid_keypoints_np[valid_keypoint_counter:valid_keypoint_counter + 1, :] - keypoints_np), axis=1,
            keepdims=False)  # M
        mask = distance_array > NMS_radius  # M

        keypoints_np = keypoints_np[mask, ...]
        sals_np = sals_np[mask]

        # increase counter
        valid_keypoint_counter += 1

    if total_num >= valid_keypoint_counter:
        return valid_keypoints_np[0:valid_keypoint_counter, :], \
                valid_sals_np[0:valid_keypoint_counter]
    else:
        return valid_keypoints_np[0:total_num, :], \
                valid_sals_np[0:total_num]


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts


def farthest_sampling(input_pts, sample_num):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(input_pts)
    
    value = int(input_pts.shape[0] / sample_num)
    pcd_sampled = o3d.geometry.PointCloud.uniform_down_sample(pcd, value)
    pcd_sampled_array = np.asarray(pcd_sampled.points)
    if pcd_sampled_array.shape[0] > sample_num:
        return pcd_sampled_array[:sample_num]
    else:
        choose_1 = np.arange(pcd_sampled_array.shape[0])
        choose_2 = np.pad(choose_1, (0, sample_num-len(choose_1)), 'wrap')
        return pcd_sampled_array[choose_2]


def max_pool_3d(kpts, score, pool_reso, total_kps): # score: B, N, dim
    kpts = kpts.unsqueeze(0)
    score = score.unsqueeze(0).unsqueeze(2)

    norm_kpts = normalize_3d_coordinate(kpts/2.0, padding=0) # B, N, 3 
    index = coordinate2index(norm_kpts, pool_reso, coord_type='3d') # B, 1, N 
    
    # fea_dim = 1
    fea = scatter_max(score.permute(0, 2, 1), index, dim_size=pool_reso**3)
    # fea = fea[0]
    # fea = fea.gather(dim=2, index=index.expand(-1, fea_dim, -1)).squeeze(0).squeeze(0)
    arg_max_index = fea[1].unique()[:-1]
    kpts = kpts[0][arg_max_index]
    score = score[0, arg_max_index, 0]

    if kpts.shape[0] > total_kps:
        sorted_idx = torch.argsort(score, 0, descending=True)
        sorted_idx = sorted_idx[0:total_kps]
        kpts_select = kpts[sorted_idx]  # Mx3
        score = score[sorted_idx]
    else:
        kpts_select = kpts
    
    return kpts_select.cpu().numpy(), score.cpu().numpy()


def matcher_b2a(descriptors_a, descriptors_b):
    #descriptors shape: (num, 128)

    if descriptors_a.dim() == 3:
        # sim = descriptors_a @ descriptors_b.permute(0, 2, 1) #shape (B, des_a.shape[0], des_a.shape[0])
        # nn12 = torch.max(sim, dim=2)[1]
        # nn21 = torch.max(sim, dim=1)[1]
        # ids1 = torch.arange(0, sim.shape[1], device=descriptors_a.device)
        # mask = (ids1 == nn21[nn12])
        # matches = torch.stack([ids1[mask], nn12[mask]])
        raise ValueError('descriptors must have 2 dims')

    elif descriptors_a.dim() == 2:
        sim = descriptors_a @ descriptors_b.t() #shape (des_a.shape[0], des_a.shape[0])
        ipdb.set_trace()
        nn21 = torch.max(sim, dim=0)[1]
    else:
        raise ValueError('descriptors must have 2 or 3 dims')
    
    return matches.t()