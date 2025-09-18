import torch.optim as optim

from common.type_utils import cfg2dict

from optim.loss.loss import Loss
from optim.optimizer.optim import get_optimizer
from optim.scheduler import get_scheduler


def build_optim(cfg, params, total_steps):
    loss = Loss(cfg) #cross-entropy
    optimizer = get_optimizer(cfg, params) #adamW
    scheduler = get_scheduler(cfg, optimizer, total_steps) #warmup_cosine
    return loss, optimizer, scheduler
