import os
import logging
import yaml
import numpy as np
import json
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0

    def empty(self):
        return len(self.history) == 0

    def update(self, val):
        self.history.append(val)
        if self.length > 0 and len(self.history) > self.length:
            del self.history[0]

        self.val = val
        self.avg = np.mean(self.history)
    

class Config(object):
    def __init__(self, config_file):

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config_path = config_file

        self.config = config
        self.config_file = config_file


def accuracy(output, target, topk=(1,), v_n=256):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = v_n

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_state(model, path, optimizer, step, tag):
    if isinstance(model, dict):
        for name, state in model.items():
            save_path = os.path.join(path, "checkpoints", "{}_{}.pth".format(name, tag))
            to_save = state.state_dict()
            torch.save(to_save, save_path)
    else:
        save_path = os.path.join(path, "checkpoints", "model_{}.pth".format(tag))
        to_save = model.state_dict()
        torch.save(to_save, save_path)
    
    torch.save(optimizer, os.path.join(path, "checkpoints", "optimizer.pth"))
    print("saving models in iter: {}".format(step))


def load_last_iter(path):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location="cpu")
        print("=> loaded last_iter={} from {}".format(checkpoint["step"], path))
        return checkpoint["step"]
    else:
        raise RuntimeError("=> no checkpoint found at {}".format(path))


def load_state(path, model, ignore=[], optimizer=None, cuda=False):
    def map_func_cuda(storage, location):
        return storage.cuda()

    def map_func_cpu(storage, location):
        return storage.cpu()

    if cuda:
        map_func = map_func_cuda
    else:
        map_func = map_func_cpu

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        keys1 = set(checkpoint["state_dict"].keys())
        keys2 = set([k for k, _ in model.named_parameters()])
        not_loaded = keys2 - keys1
        for k in not_loaded:
            print("caution: {} not loaded".format(k))

        if optimizer != None:
            assert len(ignore) == 0
            optimizer.load_state_dict(checkpoint["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    try:
                        state[k] = v.cuda()
                    except:
                        print("{} can not be set as cuda", k)

            print(
                "=> loaded checkpoint '{}' (step {})".format(path, checkpoint["step"])
            )
            return checkpoint["step"]
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)


def create_logger(name, log_file, local_rank):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)10s][line:%(lineno)4d][%(levelname)6s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


def load_model(load_weights_folder, models_to_load, net):
    
    if len(models_to_load) == 0:
        checkpoint = torch.load(load_weights_folder, lambda a,b:a)
        net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
    elif models_to_load == ['extractor']:
        pretrained_dict = torch.load(load_weights_folder)
        weights = pretrained_dict['extractor']
        net.load_state_dict(weights)
    else:
        assert os.path.isdir(load_weights_folder), \
        "Cannot find folder {}".format(load_weights_folder)
        print("loading model from folder {}".format(load_weights_folder))
        
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}_best.pth".format(n))
            model_dict = net[n].state_dict()
            pretrained_dict = torch.load(path)
            
            new_pretrained_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    new_pretrained_dict[k] = v
                elif k[7:] in model_dict:
                    new_pretrained_dict[k[7:]] = v
            model_dict.update(new_pretrained_dict)
            
            net[n].load_state_dict(model_dict)


def store_json(data, path):
    with open(path, 'w') as fw:
        json.dump(data, fw)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        return data
