import os
from time import strftime
from time import localtime

import mxnet as mx
from mxnet import ndarray as nd
import numpy as np

from lib.dataset.hm36 import JntName, HM_act_idx

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass
    if not ctx_list:
        ctx_list = [mx.cpu()]
    return ctx_list


class LOG(object):
    def __init__(self, log_path, _isDebug):
        self.isDebug = _isDebug
        self.file = None
        if not self.isDebug:
            self.file = open(log_path,'w')

    def info(self, *args):
        ctn= ''
        for item in args:
            ctn += str(item) + ' '
        if not self.isDebug:
            print(strftime("[%Y-%m-%d %H:%M:%S] ", localtime()) + ctn, file=self.file)
            self.file.flush()

        print(strftime("[%Y-%m-%d %H:%M:%S] ", localtime()) + ctn)

    def kill(self):
        if self.file:
            self.file.close()


def create_logger(cfg):
    # set up logger
    time_str = strftime('%Y-%m-%d_%H-%M')
    root_output_path = cfg.saveModel_path

    # model file save path
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)
    assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)

    # cfg_name = os.path.basename(cfg_name).split('.')[0]
    # model_path = os.path.join(root_output_path, '{}'.format(cfg_name))
    model_path = os.path.join(root_output_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    name = '{}_noise-sigma{}_gpu{}'.format(time_str, cfg.DATASET.sigma, cfg.gpu)
    if cfg.DEBUG:
        final_model_path = os.path.join(model_path,'Debug_'+name)
    else:
        final_model_path = os.path.join(model_path,'{}'.format(name))
    if not os.path.exists(final_model_path):
        os.makedirs(final_model_path)

    # train logger
    # train_log_path = os.path.join(cfg.train_log_path, '{}'.format(cfg_name))
    train_log_path = os.path.join(cfg.train_log_path)
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)

    final_train_log_path = os.path.join(train_log_path, name + '.log')
    print(final_train_log_path)
    logger = LOG(final_train_log_path, cfg.DEBUG)

    return logger, final_model_path


def LogResult(logger, config, results):
    DBsize = [item[0] for item in results]
    Loss   = [item[1] for item in results]
    Time   = [item[2] for item in results]
    MPJPE  = [item[3] for item in results]
    XYZErr = [item[4] for item in results]
    JntErr = [item[5] for item in results]

    t_size, t_error, t_loss = 0, 0, 0
    t_xyz = np.zeros(3)
    t_jnt = np.zeros(16) if not config.TEST.isPA else np.zeros(17)
    logger.info("Procrustes Analysis:", config.TEST.isPA)
    for act, size, ls, t, err, xyz_err, jnt_e  in zip(HM_act_idx, DBsize, Loss, Time, MPJPE, XYZErr, JntErr):
        t_error += size * err
        t_xyz   += size * xyz_err
        t_jnt   += size * jnt_e
        t_loss  += size * ls
        logger.info("=========================================")
        logger.info("For action          : %02d"%act)
        logger.info("Test Data Size      : %d"%size)
        logger.info("Single Forward Time : %.2f ms"% (1000*t/size))
        logger.info("Test Loss           : %.3e"%(ls))
        logger.info("MPJPE(17j)          : %.2f"%err)
        logger.info("X Y Z               : {}".format(" ".join(['%.1f'%x for x in xyz_err])))
        logger.info("-----Joint Error-----")
        for i in range(len(JntName)):
            logger.info("Joint %-10s Error: %.1f"%(JntName[i], jnt_e[i]))
        logger.info("=========================================\n")

    t_size = np.array(DBsize).sum()
    mean_error  = t_error/t_size
    mean_xyz    = t_xyz / t_size
    mean_jnterr = t_jnt / t_size
    mean_loss   = t_loss/ t_size
    logger.info("Total #Data          : %d"%t_size)
    logger.info("MEAN XYZ             : {}".format(" ".join(['%.1f'%x for x in mean_xyz])))
    logger.info("Test Loss            : %.2f"%mean_loss)
    logger.info("MPJPE(17j)           : %.2f"%mean_error)
    for i in range(len(JntName)):
        logger.info("MEAN %-10s      : %.1f"%(JntName[i], mean_jnterr[i]))


def saveModel(net, logger, config, isCKP=False, epoch=1):
    if isCKP:
        name = 'Checkpoint_epoch{}.params'.format(str(epoch))
    else:
        name = 'Model_epoch{}.params'.format(str(epoch))
    path = os.path.join(config.final_Model_path, name)
    net.collect_params().save(path)
    logger.info("Write Model/CheckPoint into", path)