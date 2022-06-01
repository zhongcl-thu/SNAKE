from .occupancy_loss import *
from .keypoint_loss import *
from .multi_losses import *

def loss_entry(config, public_params):
    return globals()[config["type"]](**config["kwargs"], **public_params)
