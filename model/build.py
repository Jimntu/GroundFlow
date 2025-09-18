import math
import torch.nn as nn
from fvcore.common.registry import Registry
from common.type_utils import cfg2dict
import torch
import os


MODEL_REGISTRY = Registry("model")


class BaseModel(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__()

    def get_opt_params(self):
        raise NotImplementedError("Function to obtain all default parameters for optimization")
    
    def count_params(self, parameters):
        tot = sum([math.prod(p.shape) for p in parameters])
        return tot
    
    def show_params_size(self, tot):
        if tot >= 1e9:
            return '{:.1f}B'.format(tot / 1e9)
        elif tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}k'.format(tot / 1e3)
        
    def save_model(self, name):
        path = os.path.join('Results',self.cfg.model.name,self.cfg.model.notes,'model_path')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), os.path.join(path, f'{name}.pt'))
        print(f'{name}.pt has already been saved in {path}')

    def load_model(self,path):
        self.load_state_dict(torch.load(path))
        print(f'{path} has already been loaded in model')


def build_model(cfg):
    model = MODEL_REGISTRY.get(cfg.model.name)(cfg)
    return model