import numpy as np
import torch
from torch.utils.data import Dataset
import os

from core.datasets.augmentation import augment
from core.datasets.smpl_model import SMPLModel, sample_vertex_from_mesh


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    colors = np.array(data[:, -1], dtype=np.int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors

def make_3d_grid(x_res, y_res, z_res, type='DHW'):
    cor_x = np.linspace(x_res[0], x_res[1], x_res[2])
    cor_y = np.linspace(y_res[0], y_res[1], y_res[2])
    cor_z = np.linspace(z_res[0], z_res[1], z_res[2])

    
    if type == 'DHW':
        Z, Y, X = np.meshgrid(cor_y, cor_z, cor_x)
        grid = np.stack((Y, Z, X)).reshape(3, -1).T
        
        tmp = np.copy(grid[:, 0])
        grid[:, 0] = grid[:, 2]
        grid[:, 2] = tmp
    
    elif type == 'HWD':
        X, Y, Z = np.meshgrid(cor_y, cor_x, cor_z)
        grid = np.stack((Y, X, Z)).reshape(3, -1).T
    
    return grid


class BaseDataset(Dataset):
    def __init__(self, config, mode):
        Dataset.__init__(self)
        
        self.mode = mode

        config_data = config["data_info"]
        config_public_params = config["common"]["public_params"]
        self.config_public_params = config_public_params
        self.config_aug = config["augmentation"]

        if mode == 'train':
            file_name = config_data["train_file"]
        elif mode == 'val':
            file_name = config_data["val_file"]
        elif mode == 'test':
            file_name = config_data["test_file"]

        with open(os.path.join(os.getcwd(), file_name), 'r') as f:
            self.filenames = f.read().splitlines()

        self.pcd_root = config_data["data_path"]
        self.input_num = config_public_params["input_pcd_num"]
        self.on_occupancy_num = config_public_params["on_occupancy_num"]
        self.off_occupancy_num = config_public_params["off_occupancy_num"]

    def __len__(self):
        return len(self.filenames)

    def get_key(self, idx, suffix):
        return self.filenames[idx] + suffix

    def __getitem__(self, idx):
        raise NotImplementedError()


class ModelNet40(BaseDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)

        config_loss = config['common'].get('loss', None)

        self.local_grid_sample = False
        if config_loss is not None:
            for item in config_loss:
                if config_loss[item]['type'] == 'Sparsity_Loss' or \
                    config_loss[item]['type'] == 'CosimLoss':
                    self.local_grid_sample = True
                
        if self.local_grid_sample:
            self.shift_grid = make_3d_grid(self.config_public_params['x_res_grid'],
                                           self.config_public_params['y_res_grid'],
                                           self.config_public_params['z_res_grid'])
            self.shift_grid /= self.config_public_params['grid_reso']
        
        self.bound_x = 0.5 * (self.config_public_params['padding'] + 1)
        self.bound_z_max = self.config_public_params.get('z_max', self.bound_x)
        self.bound_z_min = self.config_public_params.get('z_min', -self.bound_x)
        self.bound_z_max *= (self.config_public_params['padding'] + 1)
        self.bound_z_min *= (self.config_public_params['padding'] + 1)

        self.max_down_sample = self.config_aug.get('max_down_sample', 1)


    def get_pcd_full_name(self, name):
        folder = name[0:-5]
        file_name = name
        pcd_path = os.path.join(self.pcd_root, folder, file_name + '.npy')
        return pcd_path
    

    def get_input(self, index):
        # get pcd
        name = self.filenames[index]
        pcd_path = self.get_pcd_full_name(name)
        pcd_data = np.load(pcd_path)[:, :3]
        
        return pcd_data


    def normalize_pcd(self, pcd_data):
        pcd_data -= pcd_data.mean(0)

        dis = np.linalg.norm(pcd_data, axis=1)
        scale = dis.max()
        
        pcd_data /= (scale + 1e-5) # -1 ~ 1
        pcd_data /= 2 # -0.5 ~ 0.5

        return pcd_data


    def prepare_input_data(self, pcd_data, max_down_sample=1):
        if max_down_sample > 1:
            down_sample = np.random.uniform(1, max_down_sample, size=1)
        else:
            down_sample = 1
        
        rand_idcs = np.random.choice(pcd_data.shape[0], 
                                    size=int(self.input_num / down_sample), replace=True)
        pcd_data_s = pcd_data[rand_idcs]

        if rand_idcs.shape[0] < self.input_num:
            fix_idx = np.asarray(range(pcd_data_s.shape[0]))
            while pcd_data_s.shape[0] + fix_idx.shape[0] < self.input_num:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pcd_data_s.shape[0]))), 
                                            axis=0)
            random_idx = np.random.choice(pcd_data_s.shape[0], 
                                            self.input_num - fix_idx.shape[0], 
                                            replace=False)
            rand_idcs = np.concatenate((fix_idx, random_idx), axis=0)
        
            pcd_data_s = pcd_data_s[rand_idcs]
        
        return pcd_data_s


    def check_input(self, pcd, modify=True):
        valid_mask_xy = np.logical_and(np.abs(pcd)[:, 0] < self.bound_x,
                                        np.abs(pcd)[:, 1] < self.bound_x)
        valid_mask_z = np.logical_and(pcd[:, 2] < self.bound_z_max, 
                                        pcd[:, 2] > self.bound_z_min)
        valid_mask = valid_mask_xy * valid_mask_z

        if modify:
            return pcd[valid_mask]
        else:
            return pcd, valid_mask


    def prepare_occupancy_data(self, pcd_data):
        can_repeat = True if self.on_occupancy_num > pcd_data.shape[0] else False
            
        rand_idcs_on = np.random.choice(pcd_data.shape[0], 
                                        size=self.on_occupancy_num, 
                                        replace=can_repeat)
        on_surface_coords = pcd_data[rand_idcs_on]
        on_surface_labels = np.ones(self.on_occupancy_num)

        off_surface_x = np.random.uniform(-self.bound_x, self.bound_x, 
                                            size=(self.off_occupancy_num, 1))
        off_surface_y = np.random.uniform(-self.bound_x, self.bound_x, 
                                            size=(self.off_occupancy_num, 1))
        off_surface_z  = np.random.uniform(self.bound_z_min, self.bound_z_max, 
                                            size=(self.off_occupancy_num, 1))
        off_surface_coords = np.concatenate((off_surface_x, off_surface_y, off_surface_z), 
                                                axis=1)
        off_surface_labels = np.zeros(self.off_occupancy_num)
        
        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        labels = np.concatenate((on_surface_labels, off_surface_labels), axis=0)

        rix = np.random.permutation(coords.shape[0])
        coords = coords[rix]
        labels = labels[rix]

        return coords, labels


    def prepare_local_grid_data(self, pcd_data):
        rand_idcs_on = np.random.choice(pcd_data.shape[0], 
                                        size=self.config_public_params['total_grid_num'],
                                        replace=False)
        grid_xyz = pcd_data[rand_idcs_on]

        grid_xyz = np.expand_dims(grid_xyz, 1).repeat(self.shift_grid.shape[0], 1)
        grid_xyz += self.shift_grid
        grid_xyz = grid_xyz.reshape(-1, 3)

        return grid_xyz


    def __getitem__(self, index):
        pcd_data = self.get_input(index)
        pcd_data = self.normalize_pcd(pcd_data)
        pcd_data = self.check_input(pcd_data)
        
        if self.local_grid_sample:
            grid_coords = self.prepare_local_grid_data(pcd_data)
        else:
            grid_coords = None
        
        # rigid tranform to generate second view input
        pcd_data_aug, grid_coords_aug = augment(pcd_data, grid_coords, self.config_aug)
        pcd_data_aug = self.check_input(pcd_data_aug)
        
        # translate everything to the origin based on the point cloud mean
        if self.config_aug["mean_pcd"]:
            center_aug = pcd_data_aug.mean(0)
            pcd_data_aug -= center_aug
            pcd_data_aug = self.check_input(pcd_data_aug)
            if grid_coords_aug is not None:
                grid_coords_aug -= center_aug
        
        # jitter (Gaussian noise)
        sigma, clip = self.config_aug["sigma"], self.config_aug["clip"]
        jitter_pc = np.clip(sigma * np.random.randn(pcd_data_aug.shape[0], 3), 
                            -1 * clip, clip)
        pcd_data_aug_jitter = pcd_data_aug + jitter_pc

        # random choice N points
        pcd_data_s1 = self.prepare_input_data(pcd_data, 1)
        # Down-sampling and random choice 
        pcd_data_s2 = self.prepare_input_data(pcd_data_aug_jitter, self.max_down_sample)

        # get occp coords and labels
        occup_coords, labels = self.prepare_occupancy_data(pcd_data)
        occup_coords_aug, labels_aug = self.prepare_occupancy_data(pcd_data_aug)
        
        pcd_data_s1 = torch.from_numpy(pcd_data_s1).float()
        pcd_data_s2 = torch.from_numpy(pcd_data_s2).float()
        occup_coords = torch.from_numpy(occup_coords).float()
        occup_coords_aug = torch.from_numpy(occup_coords_aug).float()
        labels = torch.from_numpy(labels).long()
        labels_aug = torch.from_numpy(labels_aug).long()

        if grid_coords is not None:
            # check whether points are out of the unit volume
            _, grid_coords_mask = self.check_input(grid_coords, False)
            _, grid_coords_aug_mask = self.check_input(grid_coords_aug, False)

            grid_coords_mask = torch.from_numpy(grid_coords_mask)
            grid_coords_aug_mask = torch.from_numpy(grid_coords_aug_mask)

            grid_coords = torch.from_numpy(grid_coords).float()
            grid_coords_aug = torch.from_numpy(grid_coords_aug).float()

            out_coords = torch.cat((occup_coords, grid_coords), 0)
            out_coords_aug = torch.cat((occup_coords_aug, grid_coords_aug), 0)
        else:
            out_coords = occup_coords
            out_coords_aug = occup_coords_aug

        if grid_coords is not None:
            inputs = {'point_cloud_1': pcd_data_s1, 'coords_1': out_coords, 'occup_labels_1': labels,
                'point_cloud_2': pcd_data_s2, 'coords_2': out_coords_aug, 'occup_labels_2': labels_aug,
                'grid_coords_mask_1': grid_coords_mask, 'grid_coords_mask_2': grid_coords_aug_mask}
        else:
            inputs = {'point_cloud_1': pcd_data_s1, 'coords_1': out_coords, 'occup_labels_1': labels,
                'point_cloud_2': pcd_data_s2, 'coords_2': out_coords_aug, 'occup_labels_2': labels_aug}

        return inputs


class Match3d(ModelNet40):
    def normalize_pcd(self, pcd_data):
        
        pcd_data -= pcd_data.mean(0)
        scale = np.abs(pcd_data).max()

        pcd_data /= (scale + 1e-5) # -1 ~ 1
        pcd_data /= 2 # -0.5 ~ 0.5

        return pcd_data

    def get_pcd_full_name(self, name):
        pcd_path = os.path.join(self.pcd_root, name)
        return pcd_path


class KeypointNet(ModelNet40):
    def get_input(self, index):
        
        name = self.filenames[index]
        pcd_path = self.get_pcd_full_name(name)
        pcd_data = naive_read_pcd(pcd_path)[0]
        
        return pcd_data

    def get_pcd_full_name(self, name):
        obj_cat, ins_index = name.split('-')

        pcd_path = os.path.join(self.pcd_root, obj_cat, ins_index + '.pcd')
        return pcd_path


class SMPL(ModelNet40):
    def __init__(self, config, mode):

        config_data = config["data_info"]
        config_public_params = config["common"]["public_params"]
        self.config_public_params = config_public_params
        self.config_aug = config["augmentation"]
        
        self.smpl = SMPLModel(config_data['data_path'])
        self.len = config_data[mode + '_len']

        self.bound_x = 0.5 * (self.config_public_params['padding'] + 1)
        self.bound_z_max = self.config_public_params.get('z_max', self.bound_x)
        self.bound_z_min = self.config_public_params.get('z_min', -self.bound_x)
        self.bound_z_max *= (self.config_public_params['padding'] + 1)
        self.bound_z_min *= (self.config_public_params['padding'] + 1)

        self.max_down_sample = self.config_aug.get('max_down_sample', 1)

        config_loss = config['common'].get('loss', None)

        self.local_grid_sample = False
        if config_loss is not None:
            for item in config_loss:
                if config_loss[item]['type'] == 'Sparsity_Loss' or \
                    config_loss[item]['type'] == 'CosimLoss':
                    self.local_grid_sample = True
        
        if self.local_grid_sample:
            self.shift_grid = make_3d_grid(self.config_public_params['x_res_grid'],
                                           self.config_public_params['y_res_grid'],
                                           self.config_public_params['z_res_grid'])
            self.shift_grid /= self.config_public_params['grid_res']
        self.input_num = config_public_params["input_pcd_num"]
        self.on_occupancy_num = config_public_params["on_occupancy_num"]
        self.off_occupancy_num = config_public_params["off_occupancy_num"]


    def __len__(self):
        return self.len
    

    def get_input(self, index):
        
        pose = (np.random.rand(*self.smpl.pose_shape) - 0.5) * 0.4
        beta = (np.random.rand(*self.smpl.beta_shape) - 0.5) * 0.06
        trans = np.zeros(self.smpl.trans_shape)
        self.smpl.set_params(beta=beta, pose=pose, trans=trans)
        pc, _, _, _, _ = sample_vertex_from_mesh(self.smpl.verts, 
                                                self.smpl.faces, 
                                                num_samples=5000)
        
        return pc
