import torch
import torch.nn as nn

class Occupancy_loss (nn.Module):
    def __init__(self, weight, **kw):
        nn.Module.__init__(self)
        self.name = 'occupancy_loss'

    def forward_one(self, occ, occ_labels):
        occ_loss = -1 * (occ_labels * torch.log(occ + 1e-5) + 
                        (1 - occ_labels) * torch.log(1 - occ + 1e-5))

        return occ_loss.mean()

    def forward(self, occ1, occ2, occup_labels_1, occup_labels_2, **kw):
        B, N = occup_labels_1.shape

        occ_loss_all = 0.5 * (self.forward_one(occ1[:, :N], occup_labels_1) 
                            + self.forward_one(occ2[:, :N], occup_labels_2))

        return occ_loss_all

