import sys
import argparse
import torch
import ipdb
sys.path.append("./")

from core.utils.common import Config
from core.test import Evaluator

parser = argparse.ArgumentParser(description="3D Keypoint Detection Test and Visualization")
parser.add_argument("--test_model_root", default="", type=str)
parser.add_argument("--recover", action="store_true")
parser.add_argument("--config", default="", type=str)
parser.add_argument("--multi_gpu", action="store_true")
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--test_name', default='save_kpts', type=str)


def main():
    args = parser.parse_args()
    
    C = Config(args.config)
    args.device = 'cuda:0'
    torch.cuda.set_device(0)

    S = Evaluator(C)

    S.initialize(args)
    
    if args.test_name == 'save_kpts':
        S.save_kpts()
    elif args.test_name == 'show_occ':
        S.show_occupancy()
    elif args.test_name == 'show_sal_point':
        S.show_input_saliency()
    elif args.test_name == 'show_sal_slice':
        S.show_saliency_field_slice()


if __name__ == "__main__":
    main()
