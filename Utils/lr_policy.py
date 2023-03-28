import math


def get_lr_at_epoch(cfg, cur_epoch, old_lr):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (Config): global config object. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    # lr = get_lr_func('steps_with_relative_lrs')(cfg, cur_epoch)
    lr = get_lr_func('steps_with_relative_lrs')(cfg, cur_epoch, old_lr)

    # Perform warm up.
    if cur_epoch < cfg.warmup_epochs:
        lr_start = old_lr
        lr_end = get_lr_func('steps_with_relative_lrs')(
            cfg, cfg.warmup_epochs,old_lr
        )
        # lr_end = get_lr_func('cosine')(
        #     cfg,1
        # )
        alpha = (lr_end - lr_start) / cfg.warmup_epochs
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (Config): global config object. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return (
        0.002
        * (math.cos(math.pi * cur_epoch / (cfg.training_iterations//100)) + 1.0)
        * 0.5
    )


def lr_func_steps_with_relative_lrs(cfg, cur_epoch,old_lr):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.
    Args:
        cfg (Config): global config object. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg.LRS[ind] * old_lr


def get_step_index(cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg (Config): global config object. 
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.steps + [(cfg.training_iterations// cfg.step_iterations)]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1


def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]
