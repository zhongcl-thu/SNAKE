import numpy as np
import torch
import os

from core.utils.transform import *
from core.datasets.train_dataset import ModelNet40, make_3d_grid, naive_read_pcd
from core.datasets.smpl_model import SMPLModel, sample_vertex_from_mesh


class ModelNet40_Test(ModelNet40):
    def __init__(self, config, mode):
        super().__init__(config, mode)

        self.config = config
        self.grid_sample = config['test'].get('grid_sample', False)
        self.noise_std = self.config['test'].get('noise_std', 0)
        self.down_sample = self.config['test'].get('downsample', 1)
    
    def prepare_input_data(self, pcd_data, down_sample=1):
        input_num = int(self.input_num / down_sample)

        if pcd_data.shape[0] >= input_num:
            choice_idx = np.random.choice(pcd_data.shape[0], 
                                        input_num, replace=False)
        else:
            fix_idx = np.asarray(range(pcd_data.shape[0]))
            while pcd_data.shape[0] + fix_idx.shape[0] < input_num:
                fix_idx = np.concatenate((fix_idx, 
                                        np.asarray(range(pcd_data.shape[0]))), 
                                        axis=0)
            random_idx = np.random.choice(pcd_data.shape[0], 
                                        input_num - fix_idx.shape[0], 
                                        replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)

        pcd_data_s = pcd_data[choice_idx]
        
        return pcd_data_s

    def normalize_pcd(self, pcd_data):
        pcd_data += np.random.randn(pcd_data.shape[0], pcd_data.shape[1]) * self.noise_std
        pcd_data /= 2 # -0.5 ~ 0.5

        return pcd_data

    def prepare_occupancy_data(self, pcd_data):
        z_max = self.config_public_params.get('z_max', 0.5)
        z_min = self.config_public_params.get('z_min', -0.5)
        if self.grid_sample:
            coords = make_3d_grid([-0.5, 0.5, self.config['test']['grid_kwargs']['x_res']],
                                [-0.5, 0.5, self.config['test']['grid_kwargs']['y_res']],
                                [z_min, z_max, self.config['test']['grid_kwargs']['z_res']],
                                type='HWD')
            
        else:
            on_surface_coords = pcd_data
            off_surface_x = np.random.uniform(-0.5, 0.5, 
                                                size=(self.off_occupancy_num, 1))
            off_surface_y = np.random.uniform(-0.5, 0.5, 
                                                size=(self.off_occupancy_num, 1))
            off_surface_z  = np.random.uniform(z_min, z_max, 
                                                size=(self.off_occupancy_num, 1))
            off_surface_coords = np.concatenate((off_surface_x, 
                                                off_surface_y, 
                                                off_surface_z), 
                                                axis=1)
            coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)

        return coords

    def __getitem__(self, index):
        name = self.filenames[index]

        # get pcd1
        pcd_path_1 = os.path.join(self.pcd_root, 'original', name + '.npy')
        pcd_data_1 = np.load(pcd_path_1)[:, :3]
        pcd_data_1 = self.normalize_pcd(pcd_data_1) #add noise and normalize
        
        # get pcd2
        pcd_path_2 = os.path.join(self.pcd_root, 'rotated', name + '.npy')
        pcd_data_2 = np.load(pcd_path_2)[:, :3]
        pcd_data_2 = self.normalize_pcd(pcd_data_2) #add noise and normalize
        
        pcd_data_s1 = self.prepare_input_data(pcd_data_1, self.down_sample)
        pcd_data_s2 = self.prepare_input_data(pcd_data_2, self.down_sample)

        coords_1 = self.prepare_occupancy_data(pcd_data_s1)
        coords_2 = self.prepare_occupancy_data(pcd_data_s2)

        pcd_data_s1 = torch.from_numpy(pcd_data_s1).float()
        pcd_data_s2 = torch.from_numpy(pcd_data_s2).float()
        coords_1 = torch.from_numpy(coords_1).float()
        coords_2 = torch.from_numpy(coords_2).float()
        
        if self.config_aug["mean_pcd"]:
            center_1 = pcd_data_s1.mean(dim=0)
            coords_1 = coords_1 - center_1[None, :]
            pcd_data_s1 = pcd_data_s1 - center_1[None, :]

            center_2 = pcd_data_s2.mean(dim=0)
            coords_2 = coords_2 - center_2[None, :]
            pcd_data_s2 = pcd_data_s2 - center_2[None, :]

        inputs = {'point_cloud_1': pcd_data_s1, 'occup_coords_1': coords_1, 
                'point_cloud_2': pcd_data_s2, 'occup_coords_2': coords_2,
                'scale_1': 2.0, 'scale_2': 2.0, 'center_1': 0, 'center_2': 0}

        return inputs


class Redwood_test(ModelNet40_Test):
    def normalize_pcd(self, pcd_data):
        pcd_data += np.random.randn(pcd_data.shape[0], pcd_data.shape[1]) * self.noise_std

        center = pcd_data.mean(0)
        pcd_data -= center
        scale = np.abs(pcd_data).max()
        pcd_data /= (scale + 1e-8) # -1 ~ 1
        pcd_data /= 2 # -0.5 ~ 0.5

        return pcd_data, 2*scale, center

    def __getitem__(self, index):
        scene, name1 = self.filenames[index].split()
        
        # get pcd1
        pcd_path_1 = os.path.join(self.pcd_root, scene, name1 + '.npy')
        pcd_data_1 = np.load(pcd_path_1)[:, :3]

        pcd_data_1, scale_1, center_1 = self.normalize_pcd(pcd_data_1)
        
        pcd_data_s1 = self.prepare_input_data(pcd_data_1, self.down_sample)

        coords_1 = self.prepare_occupancy_data(pcd_data_s1)

        pcd_data_s1 = torch.from_numpy(pcd_data_s1).float()
        coords_1 = torch.from_numpy(coords_1).float()

        inputs = {'point_cloud_1': pcd_data_s1, 'occup_coords_1': coords_1, 
                'scale_1': scale_1, 'center_1': center_1}

        return inputs


class KeypointNet_test(Redwood_test):
    def __getitem__(self, index):
        name = self.filenames[index]

        obj_cat, ins_index = name.split('-')
        
        # get pcd1
        pcd_path_1 = os.path.join(self.pcd_root, obj_cat, ins_index + '.pcd')
        pcd_data_1 = naive_read_pcd(pcd_path_1)[0]

        pcd_data_1, scale_1, center_1 = self.normalize_pcd(pcd_data_1)
        
        pcd_data_s1 = self.prepare_input_data(pcd_data_1, self.down_sample)
        coords_1 = self.prepare_occupancy_data(pcd_data_s1)

        pcd_data_s1 = torch.from_numpy(pcd_data_s1).float()
        coords_1 = torch.from_numpy(coords_1).float()

        inputs = {'point_cloud_1': pcd_data_s1, 'occup_coords_1': coords_1, 
                'scale_1': scale_1, 'center_1': center_1}
        
        return inputs


class SMPL_test(ModelNet40_Test):
    def __init__(self, config, mode):

        config_data = config["data_info"]
        config_public_params = config["common"]["public_params"]
        self.config_public_params = config_public_params
        self.config_aug = config["augmentation"]
        
        self.smpl = SMPLModel(config_data['data_path'] + '/model.pkl')
        self.len = config_data[mode + '_len']

        self.bound_x = 0.5 * (self.config_public_params['padding'] + 1)
        self.bound_z_max = self.config_public_params.get('z_max', self.bound_x)
        self.bound_z_min = self.config_public_params.get('z_min', -self.bound_x)
        self.bound_z_max *= (self.config_public_params['padding'] + 1)
        self.bound_z_min *= (self.config_public_params['padding'] + 1)

        self.input_num = config_public_params["input_pcd_num"]
        self.on_occupancy_num = config_public_params["on_occupancy_num"]
        self.off_occupancy_num = config_public_params["off_occupancy_num"]

        self.config = config
        self.grid_sample = config['test'].get('grid_sample', False)
        self.noise_std = self.config['test'].get('noise_std', 0)
        self.down_sample = self.config['test'].get('downsample', 1)

        self.pose_params = np.load(config_data['data_path'] + '/pose_test_{}.npy'.format(self.len))
        self.beta_params = np.load(config_data['data_path'] + '/beta_test_{}.npy'.format(self.len))

    def __len__(self):
        return self.len
    
    def normalize_pcd(self, pcd_data):
        pcd_data += np.random.randn(pcd_data.shape[0], pcd_data.shape[1]) * self.noise_std

        center = pcd_data.mean(0)
        pcd_data -= center
        scale = np.abs(pcd_data).max()
        pcd_data /= (scale + 1e-8) # -1 ~ 1
        pcd_data /= 2 # -0.5 ~ 0.5

        return pcd_data, 2*scale, center
    
    def __getitem__(self, index):

        for i in range(2):
            pose = (self.pose_params[index, i] - 0.5) * 0.4
            beta = (self.beta_params[index, i] - 0.5) * 0.06
            trans = np.zeros(self.smpl.trans_shape)
            self.smpl.set_params(beta=beta, pose=pose, trans=trans)
            
            if i == 0:
                pc1, _, idx, u, v = sample_vertex_from_mesh(self.smpl.verts, self.smpl.faces, num_samples=self.input_num)
            else:
                pc2, _, _, _, _ = sample_vertex_from_mesh(self.smpl.verts, self.smpl.faces, rnd_idxs=idx, 
                                                                u=u, v=v, num_samples=self.input_num)
        
        pcd_data_s1, scale1, center1 = self.normalize_pcd(pc1)
        pcd_data_s2, scale2, center2 = self.normalize_pcd(pc2)

        coords_1 = self.prepare_occupancy_data(pcd_data_s1)
        coords_2 = self.prepare_occupancy_data(pcd_data_s2)

        pcd_data_s1 = torch.from_numpy(pcd_data_s1).float()
        pcd_data_s2 = torch.from_numpy(pcd_data_s2).float()
        coords_1 = torch.from_numpy(coords_1).float()
        coords_2 = torch.from_numpy(coords_2).float()

        inputs = {'point_cloud_1': pcd_data_s1, 'occup_coords_1': coords_1, 
                'point_cloud_2': pcd_data_s2, 'occup_coords_2': coords_2,
                'scale_1': scale1, 'scale_2': scale2,
                'center_1': center1, 'center_2': center2}

        return inputs


class Match3d_test(Redwood_test): 
    def prepare_input_data(self, pcd_data, down_sample=1):
        return pcd_data
    
    def prepare_occupancy_data(self, pcd_data):
        z_max = self.config_public_params.get('z_max', 0.5)
        z_min = self.config_public_params.get('z_min', -0.5)
        if self.grid_sample:
            coords = make_3d_grid([-0.5, 0.5, self.config['test']['grid_kwargs']['x_res']],
                                [-0.5, 0.5, self.config['test']['grid_kwargs']['y_res']],
                                [z_min, z_max, self.config['test']['grid_kwargs']['z_res']],
                                type='HWD')
        else:
            coords = pcd_data

        return coords