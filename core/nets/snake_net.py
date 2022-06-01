import torch
import torch.nn as nn
from torch import distributions as dist
from . import decoder
from .encoder import encoder_dict
import torch.nn.functional as F

import ipdb

# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder
}

'''based on convolutional occupancy network: 
    https://github.com/autonomousvision/convolutional_occupancy_networks'''


class SNAKE(nn.Module):
    ''' SNAKE Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, **cfg):
        super().__init__()

        encoder = cfg['encoder']
        decoder = cfg['decoder']
        dim = 3
        c_dim = cfg['c_dim']
        encoder_kwargs = cfg['encoder_kwargs']
        
        padding = cfg['padding']
        z_max = cfg['z_max']
        z_min = cfg['z_min']
        
        self.sigmoid = cfg["sigmoid"]

        decoder_occup = decoder.get('decoder_occup', None)
        decoder_keypoint = decoder.get('decoder_keypoint', None)
        
        self.parameters_to_train = []

        self.encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding, z_max=z_max, z_min=z_min,
            **encoder_kwargs)
        self.parameters_to_train += list(self.encoder.parameters())

        if decoder_occup is not None:
            self.decoder_occup = decoder_dict[decoder_occup['decoder_type']](
                dim=dim, c_dim=c_dim, padding=padding, z_max=z_max, z_min=z_min,
                **decoder_occup['decoder_kwargs']
            )
            self.parameters_to_train += list(self.decoder_occup.parameters())
        else:
            self.decoder_occup = None

        if decoder_keypoint is not None:
            self.decoder_keypoint = decoder_dict[decoder_keypoint['decoder_type']](
                dim=dim, c_dim=c_dim, padding=padding,
                **decoder_keypoint['decoder_kwargs']
            )
            self.parameters_to_train += list(self.decoder_keypoint.parameters())
        else:
            self.decoder_keypoint = None

    def forward(self, inputs, p, index=1, return_desc=False, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
        '''

        self.index = index
        c = self.encode_inputs(inputs)

        self.outputs = {}
        if self.decoder_occup is not None:
            self.decode_occup(p, c, return_desc, **kwargs)
        
        if self.decoder_keypoint is not None:
            self.decode_saliency(p, c, **kwargs)
        
        return self.outputs

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode_occup(self, p, c, return_desc=False, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        if return_desc:
            logits, desc = self.decoder_occup(p, c, return_desc, **kwargs)
        else:
            logits = self.decoder_occup(p, c, return_desc, **kwargs)
        logits = logits.squeeze(-1)

        if self.sigmoid:
            self.outputs['occ'+self.index] = torch.sigmoid(logits)
        else:
            p_r = dist.Bernoulli(logits=logits)
            self.outputs['occ'+self.index] = p_r
        
        if return_desc:
            self.outputs['desc'+self.index] = desc # F.normalize(desc, p=2, dim=1)

    def decode_saliency(self, p, c, **kwargs):
        ''' Returns saliency probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        
        logits = self.decoder_keypoint(p, c, **kwargs)
        
        if logits.shape[2] == 1:
            sal = F.softplus(logits).squeeze(-1)
            self.outputs['sal'+self.index] = sal / (sal + 1)
        elif logits.shape[2] == 2:
            self.outputs['sal'+self.index] = F.softmax(logits, dim=2)[:, :, 1:2].squeeze(-1)

