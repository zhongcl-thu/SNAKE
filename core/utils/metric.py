import numpy as np
import ipdb

def compute_rep(anc_keypoint, pos_keypoint, T_gt=None, inlier_radius=0.03):
    if anc_keypoint.shape[0] == 0 or pos_keypoint.shape[0] == 0:
        return 0
    
    if T_gt is not None:
        pos_keypoint = np.concatenate((pos_keypoint, np.ones((pos_keypoint.shape[0], 1))), 1)
        pos_keypoint = np.matmul(T_gt, pos_keypoint.T).T 
    
    anc_keypoint = np.expand_dims(anc_keypoint, 1).repeat(pos_keypoint.shape[0], 1)
    
    dist_matrix = np.linalg.norm((anc_keypoint - pos_keypoint), axis=2)
    dist = dist_matrix.min(1)

    repeatibility = np.sum(dist < inlier_radius) / dist.shape[0]

    return repeatibility
