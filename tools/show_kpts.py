import os
import sys
sys.path.append("./")

import numpy as np
from torch.utils.data import DataLoader
import argparse
import tqdm

from core.datasets.test_dataset import *
from core.utils.viz import *
from core.utils.common import Config
from tools.eval_iou import nms_usip


def main(kpts, test_data_config, save_path, test_set='KeypointNet'):
    if test_set == 'KeypointNet':
        test_dataset = KeypointNet_test(test_data_config.config, 'test')
    elif test_set == 'ModelNet40':
        test_dataset = ModelNet40_Test(test_data_config.config, 'test')

    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=2,
                                sampler = None,
                                pin_memory=False,
                                drop_last=False
                            )

    for i, input_var in enumerate(tqdm.tqdm(test_loader)):
        pcd = input_var['point_cloud_1'].cpu().numpy()[0]
        scale = input_var['scale_1'].cpu().numpy()[0]
        center = input_var['center_1'].cpu().numpy()[0]

        pcd = scale * pcd + center
        kpt = kpts[i][:, :3]

        save_path = os.path.join(save_path, '{}_{}_kp.png'.format(test_set, i))
        #color_v = [float(int('b8', 16))/255.0, float(int('f1', 16))/255.0, float(int('cc', 16))/255.0] #plane 
        color_v = [float(int('32', 16))/255.0, float(int('b8', 16))/255.0, float(int('97', 16))/255.0]
        viz_pc_keypoint(pcd, kpt, 0.01, save_path, color_v=color_v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keypoint show")
    parser.add_argument("--config_file", default='exp/KeypointNet/train0526/config_test.yaml', type=str)
    parser.add_argument("--kpts_file", default='exp/KeypointNet/train0526/test_result/keypointnet_ours.npy', type=str)
    parser.add_argument("--save_path", default='exp/KeypointNet/train0526/test_result/', type=str)
    parser.add_argument("--dataset", default='KeypointNet', type=str, 
                                    choices=['KeypointNet', 'ModelNet40'])
    
    args = parser.parse_args()

    kpts = np.load(args.kpts_file, allow_pickle=True)
    test_data_config = Config(args.config_file)

    main(kpts, test_data_config, args.save_path, args.dataset)