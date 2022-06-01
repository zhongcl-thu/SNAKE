import sys
import argparse
import os

sys.path.append("./")
import ipdb
import numpy as np
from core.utils.metric import compute_rep


def rep_and_num(kpts1, kpts2, transform_matrix, test_root, rep_thr):
    rep_all = []
    with open(test_root + '/result_rep_thr_{}.txt'.format(rep_thr), 'w', encoding='utf-8') as f:
        #for num in [4, 8, 16, 32, 64, 128, 256]:
        for num in [4, 8, 16, 32, 48, 64, 96, 128]:
            keypoint_num = []
            repeat_ratio_all = []
            print('*********** test kpts: {}  ***********'.format(num))
            
            for i in range(len(kpts1)):
                if kpts1[i].shape[0] > num:
                    k1 = kpts1[i][:num]
                else:
                    k1 = kpts1[i]
                if kpts2[i].shape[0] > num:
                    k2 = kpts2[i][:num]
                else:
                    k2 = kpts2[i]

                rep = compute_rep(k1[:, :3], k2[:, :3], 
                                transform_matrix[i][:3], 
                                rep_thr)
                #print('repeat:', rep)
                
                repeat_ratio_all.append(rep)
                keypoint_num.append(k1.shape[0])
            
            repeat_ratio_all = np.array(repeat_ratio_all)
            keypoint_num = np.array(keypoint_num)

            print('repeat mean in kpts: {}:    {}'.format(num, repeat_ratio_all.mean()))
            print('kpts mean in kpts: {}:    {}'.format(num, keypoint_num.mean()))
            rep_all.append(repeat_ratio_all.mean())
            
            f.write('test {} kpts, detect {:.4f} kpts: {:.4f}\n'.format(num, 
                                                                keypoint_num.mean(),
                                                                repeat_ratio_all.mean()))
    
        for rep in rep_all:
            f.write('{:.3f} '.format(rep))


def rep_and_repthr(kpts1, kpts2, transform_matrix, test_root, select_num=64, thresholds=[0.03, 0.04, 0.05]):
    print('*********** test kpts: {}  ***********'.format(select_num))
   
    rep_dict = {i: [] for i in thresholds}
    kp_num_dict = {i: [] for i in thresholds}
    
    for i in range(len(kpts1)):
        if kpts1[i].shape[0] > select_num:
            k1 = kpts1[i][:select_num]
        else:
            k1 = kpts1[i]
        if kpts2[i].shape[0] > select_num:
            k2 = kpts2[i][:select_num]
        else:
            k2 = kpts2[i]

        for thr in thresholds:
            if transform_matrix is None:
                rep = compute_rep(k1[:, :3], k2[:, :3], None, thr)
            else:
                rep = compute_rep(k1[:, :3], k2[:, :3], transform_matrix[i][:3], thr)
            rep_dict[thr].append(rep)
            kp_num_dict[thr].append(k1.shape[0])
    
    with open(test_root + '/result_kpts{}.txt'.format(select_num), 'w', encoding='utf-8') as f:
        for thr in thresholds:
            rep_dict[thr] = np.array(rep_dict[thr])
            kp_num_dict[thr] = np.array(kp_num_dict[thr])

            print('*********thr:', thr)
            print('repeat mean in {:.4f} kpts: {:.4f}'.format(kp_num_dict[thr].mean(), rep_dict[thr].mean()))
        
            f.write('thr {}, test {} kpts, detect {:.4f} kpts, rep {:.4f}\n'.format(thr, 
                                                            select_num,
                                                            kp_num_dict[thr].mean(),
                                                            rep_dict[thr].mean()))
        for thr in thresholds:
            f.write('{:.3f} '.format(rep_dict[thr].mean()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute repeatability")
    parser.add_argument("--dataset", default="ModelNet40", type=str)
    parser.add_argument("--test_root", default="exp/ModelNet40/train0526/test_result/noise-0-down-1-grid64-nms0.01-sal0.7-occ0.8-update10-lr0.001-pad0.125", type=str)
    parser.add_argument("--rep_thr", default=0.04, type=float)
    parser.add_argument("--keypoint_num", default=64, type=int)
    parser.add_argument('--method', default='ours', type=str)
    
    args = parser.parse_args()

    if args.dataset == 'ModelNet40':
        transform_file_name = 'data/modelnet40/modelnet40-test_rotated_numpy/transform.npy'
    elif args.dataset == 'Redwood':
        transform_file_name = 'data/redwood/numpy_gt_normal/transform.npy'
    transform_matrix = np.load(transform_file_name, allow_pickle=True)
    
    if args.dataset == 'ModelNet40':
        rep_thr_list = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        if args.method == 'ours' or args.method == 'ukpgan':
            kpts1 = np.load(args.test_root + '/kpts1.npy', allow_pickle=True)
            kpts2 = np.load(args.test_root + '/kpts2.npy', allow_pickle=True)
        else:
            kpts1 = []
            kpts2 = []
            for i in range(2468):
                kpt = np.fromfile(args.test_root + '/original/%d.bin'%i,dtype = 'float32')
                kpt = np.reshape(kpt,(-1,3))
                kpts1.append(kpt)

                kpt = np.fromfile(args.test_root + '/rotated/%d.bin'%i,dtype = 'float32')
                kpt = np.reshape(kpt,(-1,3))
                kpts2.append(kpt)
    
    elif args.dataset == 'Redwood':
        rep_thr_list = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24]
        redwood_test = {'livingroom1': 327, 'livingroom2': 199, 'office1': 231, 'office2': 183}
        pair_idx = np.loadtxt('core/datasets/split/redwood_pair.txt', dtype=np.str)
        kpts1 = []
        kpts2 = []
        num = 0
        
        for k, v in redwood_test.items():
            kpts_part = []
            if args.method == 'ours' or args.method == 'ukpgan':
                kpts_part = np.load(os.path.join(args.test_root, k, 'kpts.npy'), allow_pickle=True)
            else:
                file_num = len(os.listdir(os.path.join(args.test_root, k)))
                for i in range(file_num):
                    kpt = np.fromfile(os.path.join(args.test_root,k, '%d.bin'%i), dtype='float32')
                    kpt = np.reshape(kpt, (-1,3))
                    kpts_part.append(kpt)
                kpts_part = np.array(kpts_part)
            pair_idx_part = pair_idx[num:num+v][:, 1:].astype(np.int)
            kpts1.extend(kpts_part[pair_idx_part[:, 0]])
            kpts2.extend(kpts_part[pair_idx_part[:, 1]])
            num += v
    
    rep_and_repthr(kpts1, kpts2, transform_matrix, args.test_root, args.keypoint_num, rep_thr_list)
