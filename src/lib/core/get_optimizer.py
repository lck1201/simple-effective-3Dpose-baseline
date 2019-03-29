import mxnet as mx

def get_optimizer(cfg, ctx):
    optimizer = cfg.TRAIN.optimizer
    lr = cfg.TRAIN.lr
    lr_scheduler = mx.lr_scheduler.FactorScheduler(cfg.TRAIN.decay_steps, factor=cfg.TRAIN.decay_rate)
    optimizer_params = {
        'learning_rate': lr*len(ctx),
        'lr_scheduler': lr_scheduler 
    }
    return optimizer, optimizer_params