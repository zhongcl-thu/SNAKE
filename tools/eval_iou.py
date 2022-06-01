import argparse
import numpy as np
import os
import json
import pickle
import ipdb
import sys
sys.path.append("./")

from sklearn.metrics import pairwise_distances_argmin, pairwise_distances_argmin_min
from sklearn import neighbors
from sklearn.utils.graph import graph_shortest_path

from core.utils.viz import viz_pc_keypoint
from core.datasets.smpl_model import SMPLModel, sample_vertex_from_mesh
from core.datasets.train_dataset import naive_read_pcd


def ensure_keypoint_number(frame_keypoint_np, frame_pc_np, keypoint_num):
    if frame_keypoint_np.shape[0] == keypoint_num:
        return frame_keypoint_np
    elif frame_keypoint_np.shape[0] > keypoint_num:
        return frame_keypoint_np[np.random.choice(frame_keypoint_np.shape[0], keypoint_num, replace=False), :]
    else:
        try:
            additional_frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], keypoint_num-frame_keypoint_np.shape[0], replace=False), :]
        except:
            additional_frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], keypoint_num-frame_keypoint_np.shape[0], replace=True), :]
        frame_keypoint_np = np.concatenate((frame_keypoint_np, additional_frame_keypoint_np), axis=0)
        return frame_keypoint_np

# adapted from usip
def nms_usip(keypoints_np, sigmas_np, NMS_radius):
    '''

    :param keypoints_np: Mx3
    :param sigmas_np: M
    :return: valid_keypoints_np, valid_sigmas_np, valid_descriptors_np
    '''
    if NMS_radius < 0.01:
        return keypoints_np, sigmas_np

    valid_keypoint_counter = 0
    valid_keypoints_np = np.zeros(keypoints_np.shape, dtype=keypoints_np.dtype)
    valid_sigmas_np = np.zeros(sigmas_np.shape, dtype=sigmas_np.dtype)

    while keypoints_np.shape[0] > 0:

        max_idx = np.argmax(sigmas_np, axis=0)

        valid_keypoints_np[valid_keypoint_counter, :] = keypoints_np[max_idx, :]
        valid_sigmas_np[valid_keypoint_counter] = sigmas_np[max_idx]
        # remove the rows that within a certain radius of the selected minimum
        distance_array = np.linalg.norm(
            (valid_keypoints_np[valid_keypoint_counter:valid_keypoint_counter + 1, :] - keypoints_np), axis=1,
            keepdims=False)  # M
        mask = distance_array > NMS_radius  # M

        keypoints_np = keypoints_np[mask, ...]
        sigmas_np = sigmas_np[mask]

        # increase counter
        valid_keypoint_counter += 1

    return valid_keypoints_np[0:valid_keypoint_counter, :], \
           valid_sigmas_np[0:valid_keypoint_counter]


def eval_det_cls(pred, gt, geo_dists, dist_thresh=0.1):
    npos = 0
    fp_sum = 0
    fn_sum = 0
    for mesh_name in gt.keys():
        gt_kps = np.array(gt[mesh_name]).astype(np.int32)
        npos += len(gt_kps)
        pred_kps = np.array(pred[mesh_name]).astype(np.int32)
        fp = np.count_nonzero(np.all(geo_dists[mesh_name][pred_kps][:, gt_kps] > dist_thresh, axis=-1))
        fp_sum += fp
        fn = np.count_nonzero(np.all(geo_dists[mesh_name][gt_kps][:, pred_kps] > dist_thresh, axis=-1))
        fn_sum += fn

    return (npos - fn_sum) / np.maximum(npos + fp_sum, np.finfo(np.float64).eps)


def eval_iou(pred_all, gt_all, geo_dists, dist_thresh=0.05):
    iou = {}
    for classname in gt_all.keys():
        iou[classname] = eval_det_cls(pred_all[classname], gt_all[classname], geo_dists, dist_thresh)

    return iou


def gen_geo_dists(pc):
    graph = neighbors.kneighbors_graph(pc, 20, mode='distance', include_self=False)
    return graph_shortest_path(graph, directed=False)


name2id = {
    'airplane': '02691156',
    'chair': '03001627',
    'table': '04379243'
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute mIoU")
    parser.add_argument('--test_root', default='exp/SMPL/train0526/test_result/noise-0-down-1-grid64-nms0.1-sal0.7-occ0.8-update10-lr0.001-pad0.125/', help='data root')
    parser.add_argument('--method', type=str, default='ours')
    parser.add_argument('--dataset', type=str, default='SMPL')
    parser.add_argument('--human_annots_path', type=str, default='data/keypointnet_pcds/annotations')
    parser.add_argument('--keypointnet_data_path', type=str, default='data/keypointnet_pcds')
    args = parser.parse_args()
    
    # prepare test set
    if args.dataset == 'KeypointNet':
        with open('core/datasets/split/keypointnet_test_full.txt', 'r') as f:
            test_file_names = f.read().splitlines()
        
        test_set_num = {'airplane':205, 'chair':200, 'table':225}
        test_id_name = {'02691156':'airplane', '03001627': 'chair', '04379243':'table'}
    elif args.dataset == 'SMPL':
        pose_params = np.load('data/smpl_model/pose_test_100.npy')
        beta_params = np.load('data/smpl_model/beta_test_100.npy')
        smpl_class = SMPLModel('data/smpl_model/model.pkl')
        pc1_all = []
        pc2_all = []
        for index in range(100):
            for i in range(2):
                pose = (pose_params[index, i] - 0.5) * 0.4
                beta = (beta_params[index, i] - 0.5) * 0.06
                trans = np.zeros(smpl_class.trans_shape)
                smpl_class.set_params(beta=beta, pose=pose, trans=trans)
                
                if i == 0:
                    pc1, _, idx, u, v = sample_vertex_from_mesh(smpl_class.verts, smpl_class.faces, num_samples=2048)
                    pc1_all.append(pc1)
                else:
                    pc2, _, _, _, _ = sample_vertex_from_mesh(smpl_class.verts, smpl_class.faces, rnd_idxs=idx, 
                                                                    u=u, v=v, num_samples=2048)
                    pc2_all.append(pc2)

    # load keypoints
    if args.dataset == 'KeypointNet':
        if args.method == 'ours' or args.method == 'ukpgan':
            kpts_path = os.path.join(args.test_root, 'keypointnet_{}.npy'.format(args.method))
            # kpts have been filtered by NMS (radius=0.1)
            kpts = np.load(kpts_path, allow_pickle=True)
        else:
            kpts = []
            for name in test_file_names:
                kpt_path = os.path.join(args.test_root, '{}.bin'.format(name))
                # USIP kpts have been filtered by NMS (radius=0.1)
                kpt = np.fromfile(kpt_path, dtype='float32').reshape(kpt, (-1, 3))
                kpts.append(kpt)
            kpts = np.array(kpts)
    
    elif args.dataset == 'SMPL':
        if args.method == 'ours' or args.method == 'ukpgan':
            kpts1 = np.load(args.test_root + '/kpts1.npy', allow_pickle=True)
            kpts2 = np.load(args.test_root + '/kpts2.npy', allow_pickle=True)
        else:
            kpts1, kpts2 = [], []
            for i in range(100):
                #kpt1
                kpt = np.fromfile(args.test_root + '/tensor({})_{}.bin'.format(i,0), dtype = 'float32')
                kpt = np.reshape(kpt, (-1, 3))
                # traditional methods (fix keypoint number)
                if args.method != 'usip':
                    if kpt.shape[0] >= 20:
                        rand_idcs = np.random.choice(kpt.shape[0], size=20, replace=False)
                        kpt = kpt[rand_idcs]
                    else:
                        kpt = ensure_keypoint_number(kpt, pc1_all[i], 20)
                # else:
                #     if kpt.shape[0] >= 20:
                #         kpt = kpt[:20]
                #     else:
                #         kpt = ensure_keypoint_number(kpt, pc1_all[i], 20)
                kpts1.append(kpt)

                #kpt2
                kpt = np.fromfile(args.test_root + '/tensor({})_{}.bin'.format(i, 1),dtype = 'float32')
                kpt = np.reshape(kpt, (-1, 3))

                if args.method != 'usip':
                    if kpt.shape[0] >= 20:
                        rand_idcs = np.random.choice(kpt.shape[0], size=20, replace=False)
                        kpt = kpt[rand_idcs]
                    else:
                        kpt = ensure_keypoint_number(kpt, pc2_all[i], 20)
                # else:
                #     if kpt.shape[0] >= 20:
                #         kpt = kpt[:20]
                #     else:
                #         kpt = ensure_keypoint_number(kpt, pc2_all[i], 20)
                kpts2.append(kpt)

            kpts1, kpts2 = np.array(kpts1), np.array(kpts2)

    f = open(args.test_root + '/iou_test.txt', 'w')
    # compute
    if args.dataset == 'KeypointNet':
        num = 0
        for cat_name, cat_num in test_set_num.items():
            f.write(cat_name)
            f.write('\n')

            pred_all_iou = {cat_name: {}}
            gt_all = {cat_name: {}}

            annots = json.load(open(os.path.join(args.human_annots_path, '{}.json'.format(cat_name))))
            labels = dict([(annot['class_id']+'-'+annot['model_id'], \
                        [kp_info['pcd_info']['point_index'] for kp_info in annot['keypoints']]) \
                        for annot in annots])

            for i in range(cat_num):
                cat_id, mesh_name = test_file_names[num+i].split('-')
                cat_name = test_id_name[cat_id]
                if mesh_name not in pred_all_iou[cat_name]:
                    pred_all_iou[cat_name][mesh_name] = []

                if mesh_name not in gt_all[cat_name]:
                    gt_all[cat_name][mesh_name] = []
                    
                pc = naive_read_pcd(os.path.join(args.keypointnet_data_path, 
                                        '{}/{}.pcd'.format(cat_id, mesh_name)))[0]
                prediction = kpts[num+i][:, :3]
                
                predict_idx = pairwise_distances_argmin(prediction, pc)
                pred_all_iou[cat_name][mesh_name].extend(predict_idx)
                
                label = labels[test_file_names[num+i]]
                bin_label = np.zeros((pc.shape[0],), dtype=np.int64)
                bin_label[label] = 1

                # if i % 100 == 0:
                #     viz_pc_keypoint(pc, pc[label], 0.01)
                
                for kp in np.where(bin_label == 1)[0]:
                    gt_all[cat_name][mesh_name].append(kp)

            # compute geodesic distances between input points
            # implemented by https://github.com/qq456cvb/UKPGAN/blob/master/eval_iou.py
            BASEDIR = args.keypointnet_data_path
            if not os.path.exists(os.path.join(BASEDIR, 'cache')):
                os.makedirs(os.path.join(BASEDIR, 'cache'))
            if os.path.exists(os.path.join(BASEDIR, 'cache', '{}_geodists.pkl'.format(cat_name))):
                print('Found geodesic cache...')
                geo_dists = pickle.load(open(os.path.join(BASEDIR, 'cache', 
                                        '{}_geodists.pkl'.format(cat_name)), 'rb'))
            else:
                geo_dists = {}
                print('Generating geodesics, this may take some time...')
                for i in range(cat_num):
                    cat_id, mesh_name = test_file_names[num+i].split('-')
                    pc = naive_read_pcd(os.path.join(args.keypointnet_data_path, 
                                                '{}/{}.pcd'.format(cat_id, mesh_name)))[0]
                    geo_dists[mesh_name] = gen_geo_dists(pc).astype(np.float32)
                cat_name = test_id_name[cat_id]
                pickle.dump(geo_dists, open(os.path.join(BASEDIR, 'cache', '{}_geodists.pkl'.format(cat_name)), 'wb'))
            
            num += cat_num
            for i in range(11):
                dist_thresh = 0.01 * i
                iou = eval_iou(pred_all_iou, gt_all, geo_dists, dist_thresh=dist_thresh)

                iou_l = list(iou.values())
                s = ""
                for x in iou_l:
                    s += "{}\t".format(x)
                f.write('mIoU-{}: {}\n'.format(dist_thresh, s))
                print('mIoU-{}: {}'.format(dist_thresh, s))
                
            f.write('\n')
            
    elif args.dataset == 'SMPL':
        
        dist1_all = []
        dist2_all = []
        pred_kps_all = []
        gt_kps_all = []

        for i in range(100):
            pc1 = pc1_all[i]
            pc2 = pc2_all[i]
            
            prediction = kpts1[i][:, :3]
            if prediction.shape[0] >= 20:
                prediction = prediction[:20]
            else:
                prediction = ensure_keypoint_number(prediction, pc1, 20)

            label = kpts2[i][:, :3]
            if label.shape[0] >= 20:
                label = label[:20]
            else:
                label = ensure_keypoint_number(label, pc2, 20)
            
            predict_idx = pairwise_distances_argmin(prediction, pc1)
            pred_kps = pc1[predict_idx]
            pred_kps_all.append(pred_kps)

            label_idx = pairwise_distances_argmin(label, pc2)
            gt_kps = pc2[label_idx]
            gt_kps_all.append(gt_kps)

            pred2gt_kps = pc2[predict_idx]
            gt2pred_kps = pc1[label_idx]

            _, dist1 = pairwise_distances_argmin_min(pred2gt_kps, gt_kps)
            _, dist2 = pairwise_distances_argmin_min(gt2pred_kps, pred_kps)

            dist1_all.append(dist1)
            dist2_all.append(dist2)

        for i in range(11):
            dist_thresh = 0.01 * i
            
            npos = 0
            fp_sum = 0
            fn_sum = 0
            iou = 0

            for j in range(100):
                npos += len(gt_kps_all[j])
                fp = np.count_nonzero(dist1_all[j] > dist_thresh, axis=-1)
                fp_sum += fp
                fn = np.count_nonzero(dist2_all[j] > dist_thresh, axis=-1)
                fn_sum += fn

                iou += (npos - fn_sum) / np.maximum(npos + fp_sum, np.finfo(np.float64).eps)
            
            iou /= 100.0
            f.write('mIoU-{}: {}\n'.format(dist_thresh, iou))
            print('mIoU-{}: {}'.format(dist_thresh, iou))

    f.close()