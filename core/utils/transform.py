import numpy as np

def coordinate_ENU_to_cam_element(pc_np):
    pc_cam_np = np.copy(pc_np)
    pc_cam_np[:, 0] = pc_np[:, 0]  # x <- x
    pc_cam_np[:, 1] = -pc_np[:, 2]  # y <- -z
    pc_cam_np[:, 2] = pc_np[:, 1]  # z <- y
    return pc_cam_np


def coordinate_cam_to_ENU_element(pc_np):
    pc_enu_np = np.copy(pc_np)
    pc_enu_np[:, 0] = pc_np[:, 0]
    pc_enu_np[:, 1] = pc_np[:, 2]
    pc_enu_np[:, 2] = -pc_np[:, 1]
    return pc_enu_np