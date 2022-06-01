import torch.nn as nn

'''implemented by r2d2: https://github.com/naver/r2d2/blob/master/nets/losses.py'''

class MultiLoss(nn.Module):
    """ Combines several loss functions for convenience."""
   
    def __init__(self, loss_list):
        super(MultiLoss, self).__init__()
        self.losses = loss_list

        self.weights = []
        for loss in loss_list:
            self.weights.append(loss.weight)

    def forward(self, **variables):
        d = dict()
        cum_loss = []
        
        for weight, loss_func in zip(self.weights, self.losses):
            l = loss_func(**{k:v for k, v in variables.items()})
            
            if isinstance(l, tuple):
                loss_dict = {}
                for i, l_name in enumerate(loss_func.name):
                    loss_dict[l_name] = l[i]
                l = l[0], loss_dict
            else:
                l = l, {loss_func.name:l}
            cum_loss.append(weight * l[0])
            
            for key, value in l[1].items():
                d[key] = value
        
        d['loss'] = sum(cum_loss)
        return sum(cum_loss), d






