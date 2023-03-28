import torch
import Utils.lr_policy as lr_policy
import math


def get_epoch_lr(cur_epoch, cfg, old_lr):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cur_epoch (float): current poch id.
        cfg (Config): global config object, including the settings on 
            warm-up epochs, base lr, etc.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch, old_lr)

def set_lr(param_group, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    # for param_idx, param_group in enumerate(optimizer.param_groups):
    if "lr_reduce" in param_group.keys() and param_group["lr_reduce"]:
        # reduces the lr by a factor of 10 if specified for lr reduction
        param_group["lr"] = new_lr / 10
    else:
        param_group["lr"] = new_lr