import os
import ipdb
import tqdm
import shutil
import numpy as np
import open3d as o3d
from easydict import EasyDict as edict
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances_argmin_min

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import core.nets as nets
from core.datasets.test_dataset import *
from core.utils.match import nms
from core.utils.viz import *


class Evaluator:
    def __init__(self, C):
        self.C = C
        self.config_common = edict(C.config["common"])
        self.config_test = edict(C.config["test"])
        self.data_config = edict(self.C.config['data_info'])

        tmp = edict()
        tmp.log = {}
        self.tmp = tmp
        self.mean = lambda lis: sum(lis) / len(lis)


    def initialize(self, args):
        self.device = args.device
        self.multi_gpu = args.multi_gpu
        self.local_rank = args.local_rank
        self.test_model_root = args.test_model_root
        self.nprocs = torch.cuda.device_count()
        self.args = args
        
        self.create_model()
        self.create_dataset()
        self.create_dataloader()

        name = 'noise-{}-down-{}-grid{}-nms{}-sal{}-occ{}-update{}-lr{}-pad{}'.format(
                                                self.config_test.get('noise_std', 0),
                                                self.config_test.get('downsample', 1),
                                                self.config_test.grid_kwargs.x_res,
                                                self.config_test.nms_thr, 
                                                self.config_test.saliency_thr,
                                                self.config_test.occupancy_thr, 
                                                self.config_test.update_max, 
                                                self.config_test.optim.kwargs.lr,
                                                self.config_common.public_params.padding)
        
        self.save_root = os.path.join(args.test_model_root, 'test_result', name)
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
            shutil.copy(args.test_model_root+'/config_test.yaml', self.save_root)


    def create_model(self):
        model_path = self.data_config.get('model_path', None)
        config_public = self.config_common.public_params

        self.model = nets.model_entry(self.config_common.net, config_public)
        if model_path != None:
            try:
                self.model.load_state_dict(torch.load(os.path.join(self.test_model_root, model_path)))
            except:
                checkpoint = torch.load(os.path.join(self.test_model_root, model_path), lambda a,b:a)
                self.model.load_state_dict({k[7:]:v for k,v in checkpoint.items()})
        
        self.model.to(self.device)
        if self.multi_gpu:
            self.model = self.set_model_ddp(self.model)


    def set_model_ddp(self, m):
        m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
        return torch.nn.parallel.DistributedDataParallel(m,
                                                      device_ids=[self.local_rank],
                                                      output_device=self.local_rank,
                                                      broadcast_buffers=False)
    

    def create_dataset(self,):
        self.kp_radius = 0.01
        if self.data_config.dataset_name == 'ModelNet40':
            self.test_dataset = ModelNet40_Test(self.C.config, 'test')
        elif self.data_config.dataset_name == 'Redwood':
            self.test_dataset = Redwood_test(self.C.config, 'test')
            self.kp_radius = 0.03
        elif self.data_config.dataset_name == 'KeypointNet':
            self.test_dataset = KeypointNet_test(self.C.config, 'test')
        elif self.data_config.dataset_name == "SMPL":
            self.test_dataset = SMPL_test(self.C.config, 'test')
            self.kp_radius = 0.03
        elif self.data_config.dataset_name == "Match3d_test":
            self.test_dataset = Match3d_test(self.C.config, 'test')
            self.kp_radius = 0.03


    def create_dataloader(self):
        self.test_loader = DataLoader(
                                self.test_dataset,
                                batch_size=self.config_common.batch_size,
                                shuffle=False,
                                num_workers=self.config_common.workers,
                                sampler = None,
                                pin_memory=True,
                                drop_last=False
                            )


    def create_optimizer(self, parameters_to_train):
        '''
        optimization for refining coordinates of keypoint 
        '''
        config_optim = self.config_test.optim
        
        optim_method = config_optim.type
        if optim_method == 'Adam':
            self.optimizer = optim.Adam(parameters_to_train, 
                                    lr=config_optim.kwargs.lr)
        elif optim_method == 'SGD':
            nesterov = config_optim.get("nesterov", False)
            self.optimizer = torch.optim.SGD(
                parameters_to_train,
                config_optim.kwargs.base_lr,
                momentum=config_optim.kwargs.momentum,
                weight_decay=config_optim.kwargs.weight_decay,
                nesterov=nesterov,
            )
        else:
            raise ValueError("do not support {} optimizer".format(optim_method))


    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        if isinstance(self.model, dict):
            for m in self.model.values():
                m.eval()
        else:
            self.model.eval()


    def prepare_data(self):
        tmp = self.tmp
        for k, v in tmp.input_var.items():
            if not isinstance(v, list):
                tmp.input_var[k] = v.cuda(non_blocking=True)


    def inf_filter(self, input_pcd, input_coord, idx, filter_thr=0.5):
        '''Inference once to filter the points are likely occupied
        '''
        with torch.no_grad():
            outputs = self.model(input_pcd, input_coord, index=idx)
            if 'occ'+idx not in outputs.keys():
                outputs['occ'+idx] = torch.ones_like(outputs['sal'+idx])
            valid_initials = input_coord[outputs['occ'+idx] > filter_thr]
        
        # To align the number of filtered points in each input
        batches = torch.where(outputs['occ'+idx] > filter_thr)[0]
        batch_freq = torch.bincount(batches)
        max_num = batch_freq.max()

        input_coord_new = []
        start = 0
        for i in range(input_pcd.shape[0]):
            coord = valid_initials[start : start+batch_freq[i]]
            start += batch_freq[i]
            
            if batch_freq[i] < max_num:
                pad = coord[-1].repeat(max_num - batch_freq[i], 1)
                coord = torch.cat((coord, pad), 0)

            input_coord_new.append(coord)
        
        return torch.stack(input_coord_new), batch_freq


    def find_keypoint(self, input_var, idx, sel_strategy='best', show=False):
        '''
        input_var: point_cloud, occupancy_coordinates(on/off)
        idx: view index
        sel_strategy: keypoint selection strategy. values: ['last', 'best']. 
                      'last': select keypoints after the last optimization iteration.
                      'best': select keypoints with the highest salient scores during the whole optimizations.
                      Note that the preformance of the two strategies is similar. 
        '''
        input_pcd = input_var['point_cloud_' + idx].detach()
        input_coord = input_var['occup_coords_' + idx]
        
        input_coord, batch_freq = self.inf_filter(input_pcd, input_coord, idx)

        saliency_scores = []
        kpts = []
        
        if self.config_test.update_max > 0:
            
            input_coord.requires_grad_()
            self.create_optimizer([input_coord])

            # optimization
            for i in range(self.config_test.update_max):
                outputs = self.model(input_pcd, input_coord, index=idx)

                loss = 1 - outputs['sal'+idx].mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print(loss)

                if 'occ'+idx not in outputs.keys():
                    outputs['occ'+idx] = torch.ones_like(outputs['sal'+idx])
                
                occp_mask = outputs['occ'+idx] > self.config_test.occupancy_thr
                saliency_scores.append(outputs['sal'+idx] * occp_mask)
                kpts.append(input_coord)
        else:
            outputs = self.model(input_pcd, input_coord, index=idx)
            occp_mask = outputs['occ'+idx] > self.config_test.occupancy_thr
            saliency_scores.append(outputs['sal'+idx] * occp_mask)
            kpts.append(input_coord)
        
        kpts = torch.stack(kpts, 2)

        if sel_strategy == 'last':
            saliency_scores = saliency_scores[self.config_test.update_max-1] # B, N
            max_index = torch.ones_like(saliency_scores) * (self.config_test.update_max-1)
            max_index = max_index.long().cuda()
        elif sel_strategy == 'best':
            saliency_scores, max_index = torch.stack(saliency_scores, 2).max(2)

        with torch.no_grad():
            kpts_nms = []
            descs = []
            for j in range(saliency_scores.shape[0]):
                sal_j = saliency_scores[j, :batch_freq[j]]
                max_index_j = max_index[j, :batch_freq[j]].unsqueeze(1).unsqueeze(2)
                max_index_j = max_index_j.repeat(1, 1, 3)

                kpts_j_max = kpts[j, :batch_freq[j]].gather(dim=1, index=max_index_j).squeeze(1)
                kpts_j_select = kpts_j_max[sal_j > self.config_test.saliency_thr, :].cpu().numpy()
                score_j_select = sal_j[sal_j > self.config_test.saliency_thr].cpu().numpy()
                
                scale = input_var['scale_' + idx][j].cpu().numpy()
                center = input_var['center_' + idx][j].cpu().numpy()
                kpts_j_select = scale * kpts_j_select + center

                kpt_nms, score_nms = nms(kpts_j_select,
                                    score_j_select, 
                                    self.config_test.nms_thr,
                                    self.config_test.total_kps)
                
                if self.config_test.get('return_desc', False):
                    kpt_nms_scaled = (kpt_nms - center) / scale
                    outputs = self.model(input_pcd[j][None], torch.from_numpy(kpt_nms_scaled)[None].cuda(), 
                                            index=idx, return_desc=True)
                    
                    desc = F.normalize(outputs['desc'+idx], p=2, dim=2).cpu().numpy()
                    descs.append(desc.squeeze(0))
                
                if show:
                    print('num', kpt_nms.shape[0])
                    input_pcd_new = input_pcd[j].detach().cpu().numpy()
                    input_pcd_new = scale * input_pcd_new + center
                    viz_pc_keypoint(input_pcd_new, kpt_nms, self.kp_radius)
                    #o3d.visualization.draw_geometries([input_pcd_show, kp_show])

                kpts_nms.append(np.concatenate((kpt_nms, np.expand_dims(score_nms, 1)), 1))

                # if kpts_nms[-1].shape[0] < self.config_test.total_kps:
                #     print('********Caution!!!!! selected kpts are smaller than we need*********')
        
        return kpts_nms, descs


    def save_kpts(self,):
        self.set_eval()
        tmp = self.tmp

        kpts1_all, desc1_all = [], []
        kpts2_all, desc2_all = [], []
        
        for iteration, tmp.input_var in enumerate(tqdm.tqdm(self.test_loader)):
            self.prepare_data()
            kpts1, desc1 = self.find_keypoint(tmp.input_var, idx='1', 
                                            sel_strategy=self.config_test.sel_strategy)
            
            if self.data_config.dataset_name != 'SMPL':
                kpts1_all.extend(kpts1)
                desc1_all.extend(desc1)

                if self.data_config.dataset_name == 'ModelNet40':
                    kpts2, desc2 = self.find_keypoint(tmp.input_var, idx='2', 
                                                sel_strategy=self.config_test.sel_strategy)
                    kpts2_all.extend(kpts2)
                    desc2_all.extend(desc2)
            else:
                kpts2, desc2 = self.find_keypoint(tmp.input_var, idx='2', 
                                                sel_strategy=self.config_test.sel_strategy)
                for i in range(len(kpts1)):
                    scale1 = tmp.input_var['scale_1'][i].cpu().numpy()
                    center1 = tmp.input_var['center_1'][i].cpu().numpy()
                    pcd1_i = scale1 * tmp.input_var['point_cloud_1'][i].cpu().numpy() + center1

                    scale2 = tmp.input_var['scale_2'][i].cpu().numpy()
                    center2 = tmp.input_var['center_2'][i].cpu().numpy()
                    pcd2_i = scale2 * tmp.input_var['point_cloud_2'][i].cpu().numpy() + center2

                    # select the nearest points in the input
                    kpt1_idx = pairwise_distances_argmin(kpts1[i][:, :3], pcd1_i)
                    kpts1_all.append(pcd1_i[kpt1_idx])

                    kpt2_idx = pairwise_distances_argmin(kpts2[i][:, :3], pcd2_i)
                    kpts2_all.append(pcd2_i[kpt2_idx])
        
        kpts1_all = np.array(kpts1_all)
        
        if self.data_config.dataset_name == 'Match3d_test':
            np.save(self.save_root + '/kpts.npy', kpts1_all)
            
            if self.config_test.get('return_desc', False):
                desc1_all = np.array(desc1_all)
                np.save(self.save_root + '/desc.npy', desc1_all)

        elif self.data_config.dataset_name == 'ModelNet40' or \
            self.data_config.dataset_name == 'SMPL':
            kpts2_all = np.array(kpts2_all)
            np.save(self.save_root + '/kpts1.npy', kpts1_all)
            np.save(self.save_root + '/kpts2.npy', kpts2_all)
        elif self.data_config.dataset_name == 'Redwood':
            num = 0
            redwood_scene = {'livingroom1': 57, 'livingroom2': 47, 'office1': 53, 'office2': 50}
            for k,v in redwood_scene.items():
                if not os.path.exists(os.path.join(self.save_root, k)):
                    os.makedirs(os.path.join(self.save_root, k))
                np.save(os.path.join(self.save_root, k, 'kpts.npy'), kpts1_all[num: num+v])
                num += v
        elif self.data_config.dataset_name == 'KeypointNet':
            np.save(self.save_root + '/keypointnet_ours.npy', kpts1_all)
        else:
            raise ValueError


    def show_reconstruction(self,):
        '''show object shape by marching cube
        '''
        self.set_eval()
        tmp = self.tmp

        with torch.no_grad():
            for i, tmp.input_var in enumerate(tqdm.tqdm(self.test_loader)):
                
                self.prepare_data()

                outputs = self.model(tmp.input_var['point_cloud_1'], 
                                    tmp.input_var['occup_coords_1'], index='1')

                pcd_data = tmp.input_var['point_cloud_1'].cpu().numpy()[0]

                pred_occ = outputs['occ1'].cpu().numpy()[0] #logits.
                #pred_occ = np.log(pred_occ + 1e-8) - np.log(1 - pred_occ + 1e-8)
                pred_occ = pred_occ.reshape(self.config_test.grid_kwargs.x_res, 
                                            self.config_test.grid_kwargs.y_res, 
                                            self.config_test.grid_kwargs.z_res
                                            )
                if pred_occ.shape[2] < pred_occ.shape[0]:
                    pred_occ = np.concatenate((pred_occ, 
                                    -20 * np.ones((pred_occ.shape[0], pred_occ.shape[1], 
                                                pred_occ.shape[0] - pred_occ.shape[2]))), 2)
                mesh = extract_mesh(pred_occ, threshold=0.4)
                mesh.show()
                inputs_path = os.path.join(self.save_root, '{}.ply'.format(i))
                mesh.export(inputs_path)


    def show_input_saliency(self,):
        '''show saliency of each input point
        '''
        self.set_eval()
        tmp = self.tmp

        with torch.no_grad():
            for i, tmp.input_var in enumerate(tqdm.tqdm(self.test_loader)):
                self.prepare_data()

                outputs = self.model(tmp.input_var['point_cloud_1'], 
                                    tmp.input_var['occup_coords_1'], index='1')
                
                colors = np.zeros((outputs['sal1'][0].shape[0], 3))
                pred_sal = outputs['sal1'].cpu().numpy()[0]
                colors[:, 0] = pred_sal
                
                show_pcd = make_o3d_pcd(tmp.input_var['occup_coords_1'].cpu().numpy()[0],
                                        colors)
                
                o3d.visualization.draw_geometries([show_pcd]) #draw_geometries
    
    
    def show_saliency_field_slice(self, projection=1):
        self.set_eval()
        tmp = self.tmp

        with torch.no_grad():
            for i, tmp.input_var in enumerate(tqdm.tqdm(self.test_loader)):
                self.prepare_data()

                shift = make_3d_grid([-0.5, 0.5, 150],
                                    [-0.5, 0.5, 150],
                                    [-0.5, 0.5, 150])
                outputs = self.model(tmp.input_var['point_cloud_1'], 
                        torch.from_numpy(shift).unsqueeze(0).cuda(), index='1')
                sal1 = outputs['sal1'][0].cpu().numpy().reshape(150, 150, 150) #DHW

                # select one of [0, 1, 2] to choose a projection direction.
                sal1_proj = np.max(sal1, projection) 

                plt.axis('off')
                plt.imshow(sal1_proj, cmap='Reds')

                fig = plt.gcf()
                #fig.set_size_inches(7.0/3, 7.0/3) #dpi = 300, output = 700*700 pixels
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0, 0)
                fig.savefig(os.path.join(self.save_root, '{}_smpl_field.png'.format(i)), 
                            format='png', transparent=True, dpi=300, pad_inches = 0)
                plt.show()