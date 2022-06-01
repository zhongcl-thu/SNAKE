import sys
import argparse

sys.path.append("./")
from core.utils.common import Config
from core.solvers import solver_entry
import torch
import ipdb

parser = argparse.ArgumentParser(description="3D Keypoint Detection Training")
parser.add_argument("--load-path", default="", type=str)
parser.add_argument("--recover", action="store_true")
parser.add_argument("--config", default="", type=str)
parser.add_argument("--multi_gpu", action="store_true")
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

def main():

    args = parser.parse_args()
    
    C = Config(args.config)
    
    if args.multi_gpu:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        args.device = device
        args.local_rank = local_rank
    else:
        args.device = 'cuda:0'
        torch.cuda.set_device(0)

    S = solver_entry(C)

    S.initialize(args)

    S.run()


if __name__ == "__main__":
    main()
