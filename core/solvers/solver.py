import os
import time
import random
import warnings
from collections import defaultdict
from tqdm import tqdm
import ipdb

from easydict import EasyDict as edict
from tensorboardX import SummaryWriter

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from core.datasets.train_dataset import *
import core.nets as nets
import core.losses as losses
from core.utils.common import (
    AverageMeter,
    load_state,
    load_last_iter,
    save_state,
    create_logger
)


class Solver:
    def __init__(self, C):
        self.C = C
        config_common = edict(C.config["common"])
        self.config_common = config_common
        self.save_path = config_common.get("save_path", os.path.dirname(C.config_file))

        self.data_config = edict(self.C.config['data_info'])
        
        self.last_iter = -1
        self.last_state_dict = {}
        self.last_optim_state_dict = {}
        self.last_save_iter = -1

        tmp = edict()
        tmp.log = {}
        self.tmp = tmp
        self.mean = lambda lis: sum(lis) / len(lis)

    def initialize(self, args):
        self.local_rank = args.local_rank
        self.device = args.device
        self.multi_gpu = args.multi_gpu
        self.nprocs = torch.cuda.device_count()
        if self.local_rank in [-1, 0]:
            if not os.path.exists("{}/events".format(self.save_path)):
                os.makedirs("{}/events".format(self.save_path))
            if not os.path.exists("{}/logs".format(self.save_path)):
                os.makedirs("{}/logs".format(self.save_path))
            if not os.path.exists("{}/checkpoints".format(self.save_path)):
                os.makedirs("{}/checkpoints".format(self.save_path))

            self.train_logger = SummaryWriter("{}/events/train".format(self.save_path))
            self.val_logger = SummaryWriter("{}/events/val".format(self.save_path))
        
        self.logger = create_logger(
            "global_logger", "{}/logs/log.txt".format(self.save_path), 
            self.local_rank)
        
        # specific seed
        if self.config_common.get("deterministic", "True"):
            seed = self.config_common.random_seed + args.local_rank
            cudnn.deterministic = True
            cudnn.benchmark = False
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        self.create_model()
        self.create_loss()
        self.create_dataset()
        self.create_optimizer()
        
        # recover last training
        if args.recover:
            self.last_iter = load_last_iter(args.load_path)
            self.last_iter -= 1
        self.load_args = args
        self.create_dataloader()
        self.create_lr_scheduler()

    def create_dataset(self):
        if self.data_config.dataset_name == 'ModelNet40':
            self.train_dataset = ModelNet40(self.C.config, 'train')
            self.val_dataset = ModelNet40(self.C.config, 'val')
        elif self.data_config.dataset_name == "match3d":
            self.train_dataset = Match3d(self.C.config, 'train')
            self.val_dataset = Match3d(self.C.config, 'val')
        elif self.data_config.dataset_name == "KeypointNet":
            self.train_dataset = KeypointNet(self.C.config, 'train')
            self.val_dataset = KeypointNet(self.C.config, 'val')
        elif self.data_config.dataset_name == "SMPL":
            self.train_dataset = SMPL(self.C.config, 'train')
            self.val_dataset = SMPL(self.C.config, 'val')

    def create_dataloader(self):
        if self.multi_gpu:
            self.train_sampler = DistributedSampler(self.train_dataset)
            self.val_sampler = None
            train_shuffle = False
        else:
            self.train_sampler = None
            self.val_sampler = None
            train_shuffle = True
        
        self.train_loader = DataLoader(
                                self.train_dataset,
                                batch_size=self.config_common.batch_size,
                                shuffle=train_shuffle,
                                num_workers=self.config_common.workers,
                                pin_memory=True,
                                sampler = self.train_sampler
                            )
        
        self.val_loader = DataLoader(
                                self.val_dataset,
                                batch_size=self.config_common.batch_size,
                                shuffle=False,
                                num_workers=self.config_common.workers,
                                sampler = self.val_sampler,
                                pin_memory=True,
                                drop_last=True
                            )

    def create_model(self):
        model_path = self.data_config.get('model_path', None)
        config_public = self.config_common.public_params

        self.model = nets.model_entry(self.config_common.net, config_public)
        self.parameters_to_train = self.model.parameters_to_train
        if model_path != None:
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.to(self.device)
        if self.multi_gpu:
            self.model = self.set_model_ddp(self.model)
    
    def set_model_ddp(self, m):
        m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
        return torch.nn.parallel.DistributedDataParallel(m,
                                                      device_ids=[self.local_rank],
                                                      output_device=self.local_rank,
                                                      broadcast_buffers=False)

    def create_loss(self):
        config_loss = self.config_common.loss
        config_public = self.config_common.public_params
        loss_list = []

        for item in config_loss:
            loss_list.append(losses.loss_entry(config_loss[item], config_public))
        
        self.multiloss = losses.MultiLoss(loss_list)
        for l in self.multiloss.losses:
            l.to(self.device)

    def create_optimizer(self):
        config_optim = self.config_common.optim
        
        optim_method = config_optim.type
        if optim_method == 'Adam':
            self.optimizer = optim.Adam(self.parameters_to_train, 
                                    lr=self.config_common.lr_scheduler.kwargs.base_lr, 
                                    weight_decay=config_optim.kwargs.weight_decay)
        elif optim_method == 'SGD':
            nesterov = config_optim.get("nesterov", False)
            self.optimizer = torch.optim.SGD(
                self.parameters_to_train,
                self.config_common.lr_scheduler.kwargs.base_lr,
                momentum=config_optim.momentum,
                weight_decay=config_optim.weight_decay,
                nesterov=nesterov,
            )
        else:
            raise ValueError("do not support {} optimizer".format(optim_method))

    def create_lr_scheduler(self):
        config_lr = self.config_common.lr_scheduler
        if config_lr.type == 'Step': #StepLR
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                        config_lr.kwargs.lr_step_size, 
                                                        config_lr.kwargs.lr_mults)
        else:
            raise ValueError("do not support {} lr_scheduler".format(config_lr.type))

    def tfboard_logging(self, mode='train'):
        '''tensorboard logging'''
        tmp = self.tmp
        
        if mode == 'train':
            for l, v in tmp.log.items():
                if self.local_rank in [-1, 0]:
                    self.train_logger.add_scalar("{}".format(l), 
                                                v.avg, 
                                                tmp.current_step)
        else:
            for l, v in tmp.stats_val.items():
                if self.local_rank in [-1, 0]:
                    self.val_logger.add_scalar("{}".format(l), 
                                            self.mean(v).item(), 
                                            tmp.current_step)

    def logging(self, mode='train'):
        '''logging to logs/log.txt'''
        
        tmp = self.tmp
        config = self.config_common
        if mode == 'train':
            self.logger.info(
                "epoch:[{0}/{1}]  "
                "Iter:{2}  "
                "Time:{batch_time.avg:.2f} ({data_time.avg:.2f})  "
                "TotalLoss:{totalloss:.3f}  "
                "occupancy_loss:{occupancy_loss:.3f}  "
                "repeat_loss:{repeat_loss:.3f}  "
                "sparsity_loss:{sparsity_loss:.3f}  "
                "surface_loss:{surface_loss:.3f}  "
                "precision:{precision:.3f}  "
                "recall:{recall:.3f}  ".format(
                    tmp.epoch,
                    config.max_epoch,
                    tmp.current_step,
                    batch_time=tmp.vbatch_time,
                    data_time=tmp.vdata_time,
                    totalloss=float(tmp.log['loss'].avg),
                    occupancy_loss=float(tmp.log['occupancy_loss'].avg) \
                        if tmp.log.get('occupancy_loss', None) is not None else 0,
                    repeat_loss=float(tmp.log['repeatability_loss'].avg) \
                                if tmp.log.get('repeatability_loss', None) is not None else 0,
                    sparsity_loss=float(tmp.log['sparsity_loss'].avg) \
                                if tmp.log.get('sparsity_loss', None) is not None else 0,
                    surface_loss=float(tmp.log['surface_loss'].avg) \
                                if tmp.log.get('surface_loss', None) is not None else 0,
                    precision=float(tmp.log['precision'].avg),
                    recall=float(tmp.log['recall'].avg),
                )
            )
        else:
            self.logger.info(
                "**********Validation: \t"
                "TotalLoss:{totalloss:.3f}  "
                "occupancy_loss:{occupancy_loss:.3f}  "
                "repeat_loss:{repeat_loss:.3f}  "
                "sparsity_loss:{sparsity_loss:.3f}  "
                "surface_loss:{surface_loss:.3f}  "
                "precision:{precision:.3f}  "
                "recall:{recall:.3f}  ".format(
                    totalloss=float(self.mean(tmp.stats_val['loss'])),
                    occupancy_loss=float(self.mean(tmp.stats_val['occupancy_loss'])) \
                                if tmp.stats_val.get('occupancy_loss', None) is not None else 0,
                    repeat_loss=float(self.mean(tmp.stats_val['repeatability_loss'])) \
                                if tmp.stats_val.get('repeatability_loss', None) is not None else 0,
                    sparsity_loss=float(self.mean(tmp.stats_val['sparsity_loss'])) \
                                if tmp.stats_val.get('sparsity_loss', None) is not None else 0,
                    surface_loss=float(self.mean(tmp.stats_val['surface_loss'])) \
                                if tmp.stats_val.get('surface_loss', None) is not None else 0,
                    precision=float(self.mean(tmp.stats_val['precision'])),
                    recall=float(self.mean(tmp.stats_val['recall'])),
                )
            )

    def load(self, args):
        if args.load_path == "":
            return
        if args.recover:
            self.last_iter = load_state(
                args.load_path, self.model, optimizer=self.optimizer
            )
            self.last_iter -= 1
        else:
            load_state(args.load_path, self.model)

    def pre_run(self):
        tmp = self.tmp
        tmp.vbatch_time = AverageMeter(self.config_common.print_freq)
        tmp.vdata_time = AverageMeter(self.config_common.print_freq)
        self.set_train()
    
    def reduce_mean(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.nprocs
        return rt

    def set_train(self):
        """Convert all models to training mode
        """
        if isinstance(self.model, dict):
            for m in self.model.values():
                m.train()
        else:
            self.model.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        if isinstance(self.model, dict):
            for m in self.model.values():
                m.eval()
        else:
            self.model.eval()

    def prepare_data(self):
        """to cuda
        """
        tmp = self.tmp
        for k, v in tmp.input_var.items():
            if not isinstance(v, list):
                tmp.input_var[k] = v.cuda(non_blocking=True) 

    def analyse_result(self, details, outputs2):
        """to compute precision and recall of the occupancy precition 
        """
        occ_labels = self.tmp.input_var['occup_labels_2']
        B, N = occ_labels.shape

        if 'occ2' in outputs2.keys():
            prediction = torch.where(outputs2['occ2'][:, :N] > 0.8, 1.0, 0.0)
            precision = (prediction == occ_labels).sum().float() / len(occ_labels.view(-1))
            pred_positive = (prediction == occ_labels) * (occ_labels == True)
            recall = pred_positive.sum().float() / (occ_labels == True).sum().float()
        else:
            precision = torch.tensor(0.0).cuda()
            recall = torch.tensor(0.0).cuda()

        details['precision'] = precision
        details['recall'] = recall

    def forward(self):
        tmp = self.tmp

        outputs1 = self.model(tmp.input_var['point_cloud_1'], 
                                tmp.input_var['coords_1'], 
                                index='1')
        outputs2 = self.model(tmp.input_var['point_cloud_2'], 
                                tmp.input_var['coords_2'], 
                                index='2')

        allvars = dict(**outputs1, **outputs2)
        
        l, details = self.multiloss(**allvars, **tmp.input_var)

        self.analyse_result(details, outputs2)

        return l, details, allvars

    def run(self):
        config = self.config_common
        tmp = self.tmp

        self.pre_run()

        end = time.time()

        best_loss_mean = np.inf

        for epoch in range(config.max_epoch):
            tmp.epoch = epoch
            self.logger.info(f"\n>> Starting epoch {epoch}...")

            if self.multi_gpu:
                self.train_sampler.set_epoch(epoch)

            for i, tmp.input_var in enumerate(self.train_loader):
                tmp.vdata_time.update(time.time() - end)

                self.prepare_data() # tocuda

                tmp.current_step = self.last_iter + i + 1
                self.lr_scheduler.step(epoch)
                tmp.current_lr = self.lr_scheduler.get_last_lr()[0]
                
                tmp.loss, tmp.details, _ = self.forward()
                
                if np.isnan(tmp.loss.item()):
                    self.logging.error('Loss is NaN')
                    ipdb.set_trace()
                    continue
                
                if self.multi_gpu:
                    torch.distributed.barrier()
                for k, v in tmp.details.items():
                    if self.multi_gpu:
                        v = self.reduce_mean(v)
                    
                    try:
                        tmp.log[k].update(v.item())
                    except:
                        tmp.log[k] = AverageMeter(self.config_common.print_freq)
                        tmp.log[k].update(v.item())

                self.optimizer.zero_grad()
                tmp.loss.backward()
                self.optimizer.step()

                tmp.vbatch_time.update(time.time() - end)
                end = time.time()

                if tmp.current_step % config.print_freq == 0:
                    self.tfboard_logging(mode='train')
                    self.logging()
                
                if tmp.current_step % (config.val_freq) == 0:
                    if self.val_loader != None:
                        val_loss_mean = self.val()
                        self.set_train()
                        
                        if self.local_rank in [-1, 0]:
                            if val_loss_mean < best_loss_mean:
                                save_state(self.model, self.save_path, 
                                                self.optimizer.state_dict(), 
                                                tmp.current_step, tag='best')

                                best_loss_mean = val_loss_mean

                    if self.multi_gpu:
                        torch.distributed.barrier()

            if (
                config.save_epoch_interval > 0 and epoch>10
                and (tmp.epoch+1) % config.save_epoch_interval == 0
            ):
                if self.local_rank in [-1, 0]:
                    save_state(self.model, self.save_path, 
                                self.optimizer.state_dict(), 
                                tmp.current_step, tag=tmp.epoch + 1)

            self.last_iter = tmp.current_step 

    def val(self,):
        """
        Validate the model on a single minibatch
        """
        self.set_eval()
        
        tmp = self.tmp
        stats_val = defaultdict(list)

        with torch.no_grad():
            for tmp.input_var in tqdm(self.val_loader):
                
                self.prepare_data()
                
                loss, val_details, _ = self.forward()
                
                for key, value in val_details.items():
                    stats_val[key].append(value)

        tmp.stats_val = stats_val
        
        self.tfboard_logging(mode='val')
        self.logging(mode='val')

        return self.mean(tmp.stats_val['loss'])