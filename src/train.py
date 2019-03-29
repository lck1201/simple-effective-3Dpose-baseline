import pprint

import mxnet as mx
from mxnet import gluon
from mxnet import init

from lib.core.get_optimizer import *
from lib.core.metric import MPJPEMetric
from lib.core.loss import MeanSquareLoss
from lib.core.loader import JointsDataIter
from lib.network import get_net
from lib.net_module import *
from lib.utils import *
from lib.dataset.hm36 import hm36

from config import config, gen_config

def main():
    # Parse config and mkdir output
    logger, final_Model_path = create_logger(config)
    config.final_Model_path = final_Model_path
    gen_config(os.path.join(final_Model_path, 'hyperParams.yaml'))
    logger.info('\nTraining config:{}\n'.format(pprint.pformat(config)))

    # define context
    if config.useGPU:
        ctx = [mx.gpu(int(i)) for i in config.gpu.split(',')]
    else:
        ctx = mx.cpu()
    logger.info("Using context:", ctx)

    # dataset, generate trainset/ validation set
    train_imdbs = []
    valid_imdbs = []
    for i in range(len(config.DATASET.train_image_set)):
        logger.info("Construct Dataset:", config.DATASET.dbname[i], ", Dataset Path:", config.DATASET.dataset_path[i])
        train_imdbs.append(eval(config.DATASET.dbname[i])(config.DATASET.train_image_set[i],
                                                          config.DATASET.root_path[i],
                                                          config.DATASET.dataset_path[i]))
        valid_imdbs.append(eval(config.DATASET.dbname[i])(config.DATASET.valid_image_set[i],
                                                          config.DATASET.root_path[i],
                                                          config.DATASET.dataset_path[i],
                                                          config.final_Model_path))
    data_names  = ['hm36data']
    label_names = ['hm36label']
    train_data_iter = JointsDataIter(train_imdbs[0], runmode=0,
                                    data_names = data_names, label_names=label_names,
                                    shuffle=config.TRAIN.SHUFFLE, batch_size=len(ctx)*config.TRAIN.batchsize, logger=logger)
    valid_data_iter = JointsDataIter(valid_imdbs[0], runmode=1,
                                    data_names = data_names, label_names=label_names,
                                    shuffle=False, batch_size=len(ctx)*config.TEST.batchsize, logger=logger)

    assert train_data_iter.get_meanstd()['mean3d'].all() == valid_data_iter.get_meanstd()['mean3d'].all()

    # network
    net = get_net(config)
    if config.resume:
        ckp_path = os.path.join(config.resumeckp)
        net.collect_params().load(ckp_path, ctx=ctx)
    else:
        net.initialize(init=init.MSRAPrelu(), ctx=ctx)

    if config.NETWORK.hybrid:
        net.hybridize()

    logger.info(net)

    # define loss and metric
    mean3d = train_data_iter.get_meanstd()['mean3d']
    std3d  = train_data_iter.get_meanstd()['std3d']
    train_metric = MPJPEMetric('train_metric', mean3d, std3d)
    eval_metric  = MPJPEMetric('valid_metric', mean3d, std3d)
    loss         = MeanSquareLoss()

    # optimizer
    optimizer, optimizer_params = get_optimizer(config, ctx)

    # train and valid
    TrainDBsize = train_data_iter.get_size()
    ValidDBsize = valid_data_iter.get_size()
    logger.info("Train DB size:", TrainDBsize, "Valid DB size:",ValidDBsize)

    if not isinstance(train_data_iter, mx.io.PrefetchingIter):
        train_data_iter = mx.io.PrefetchingIter(train_data_iter)

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    for epoch in range(config.TRAIN.begin_epoch, config.TRAIN.end_epoch):
        trainNet(net, trainer, train_data_iter, loss, train_metric, epoch, config, logger=logger, ctx=ctx)
        validNet(net, valid_data_iter, loss, eval_metric, epoch, config, logger=logger, ctx=ctx)

    logger.kill()

if __name__ == '__main__':
    main()