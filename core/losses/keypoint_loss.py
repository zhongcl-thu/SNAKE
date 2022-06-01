import torch
import torch.nn as nn
import torch.nn.functional as F


class CosimLoss(nn.Module):
    """ 
    Try to make the saliency field repeatable from one point cloud to the other.
    """
    def __init__(self, weight, x_res_grid, y_res_grid, z_res_grid, 
                        cos_point_ratio=0.9, **kw):
        nn.Module.__init__(self)
        self.name = 'repeatability_loss'
        self.weight = weight
        self.point_num_in_grid = x_res_grid[2] * y_res_grid[2] * z_res_grid[2]
        self.cos_point_ratio = cos_point_ratio

    def forward_one(self, sal, B, grid_coords_mask):
        sal = sal.reshape(B, -1, self.point_num_in_grid)
        sal_norm = F.normalize(sal, p=2, dim=2)
        
        grid_coords_mask = grid_coords_mask.reshape(B, -1, self.point_num_in_grid)
        cosim_loss_mask = (torch.sum(grid_coords_mask, 2) / self.point_num_in_grid) \
                            > self.cos_point_ratio

        return sal_norm, cosim_loss_mask

    def forward(self, sal1, sal2, occup_labels_1, grid_coords_mask_1, 
                    grid_coords_mask_2, **kw):

        B, N = occup_labels_1.shape 

        # the first N points are used to compute occpupancy loss
        sal1_norm, cosim_loss_mask1 = self.forward_one(sal1[:, N:], B, grid_coords_mask_1)
        sal2_norm, cosim_loss_mask2 = self.forward_one(sal2[:, N:], B, grid_coords_mask_2)
        
        cosim_loss_mask1 *= cosim_loss_mask2
        
        if cosim_loss_mask1.sum() > 0:
            cosim = (sal1_norm * sal2_norm).sum(dim=2)[cosim_loss_mask1]
            return (1 - cosim.mean()) * self.weight
        else:
            return torch.tensor(0).to(cosim_loss_mask1.device)


class Sparsity_Loss(nn.Module):
    """ 
    Try to make the saliency field locally sparse.
    """
    def __init__(self, weight, x_res_grid, y_res_grid, z_res_grid, occp_thr=0.5,
                 occp_point_ratio=0.1, inlier_point_ratio=0.8, **kw):
        nn.Module.__init__(self)
        self.name = 'sparsity_loss'
        self.weight = weight

        self.point_num_in_grid = x_res_grid[2] * y_res_grid[2] * z_res_grid[2]
        self.z_res_grid = z_res_grid
        self.y_res_grid = y_res_grid
        self.x_res_grid = x_res_grid

        self.occp_thr = occp_thr
        self.occp_point_ratio = occp_point_ratio
        self.inlier_point_ratio = inlier_point_ratio
    

    def forward_one(self, sal, occ_new, grid_coords_mask):
        B, N = sal.shape
        sal_new = sal.clone() 

        grid_coords_mask = grid_coords_mask.reshape(B, -1, self.point_num_in_grid)
        valid_grid_mask = (torch.sum(grid_coords_mask, 2) / self.point_num_in_grid) \
                            > self.inlier_point_ratio
        
        sal_new = sal_new.reshape(B, -1, self.point_num_in_grid)
        occ_new = occ_new.reshape(B, -1, self.point_num_in_grid)
        
        occp_pts_bool = (occ_new >= self.occp_thr)
        occp_pts_sum = torch.sum(occp_pts_bool, 2).float()
        valid_grid_mask_plus = (occp_pts_sum / self.point_num_in_grid) > self.occp_point_ratio

        sal_new[occ_new < self.occp_thr] = 0
        
        sparsity_loss = (1 - (sal_new.max(2)[0] - sal_new.sum(2) / (occp_pts_sum + 1e-5)))

        valid_grid_mask *= valid_grid_mask_plus
        if valid_grid_mask.sum() > 0:
            return (sparsity_loss[valid_grid_mask]).mean()
        else:
            return torch.tensor(0).to(sparsity_loss.device)


    def forward(self, sal1, sal2, occup_labels_1, occ1, occ2, 
                    grid_coords_mask_1, grid_coords_mask_2, **kw):

        occ1 = occ1.detach()
        occ2 = occ2.detach()

        B, N = occup_labels_1.shape
        
        loss = 0.5 * (self.forward_one(sal1[:, N:], occ1[:, N:], grid_coords_mask_1) 
                        + self.forward_one(sal2[:, N:], occ2[:, N:], grid_coords_mask_2))
        
        return loss * self.weight


class Surface_Loss(nn.Module):
    """ 
    Try to enforce the keypoint scatter on the underlying surface of the object/scene.
    """
    def __init__(self, weight, **kw):
        nn.Module.__init__(self)
        self.name = 'surface_loss'
        self.weight = weight


    def forward(self, sal1, sal2, occ1, occ2, occup_labels_1,
                    grid_coords_mask_1, grid_coords_mask_2, **kw):

        occ1 = occ1.detach()
        occ2 = occ2.detach()
        
        sal_mask1 = torch.cat((torch.ones_like(occup_labels_1).bool(), grid_coords_mask_1), 1)
        sal_mask2 = torch.cat((torch.ones_like(occup_labels_1).bool(), grid_coords_mask_2), 1)

        loss = 0.5 * (((1 - occ1)*sal1)[sal_mask1].mean() + 
                        ((1 - occ2)*sal2)[sal_mask2].mean())
        
        return loss * self.weight