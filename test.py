import sys
import pickle
import pprint

import mxnet as mx

from lib.core.metric import *
from lib.core.loss import MeanSquareLoss
from lib.core.loader import JointsDataIter
from lib.network import get_net
from lib.utils import *
from lib.dataset.hm36 import hm36, HM_act_idx
from lib.net_module import TestNet

from config import config, update_config, gen_config, s_args

def main():
    # Parse config and mkdir output
    if not os.path.exists(s_args.model):
        print("Model doesn't exist!!!")

    yamlPath = os.path.join(os.path.dirname(s_args.model), 'hyperParams.yaml')
    if os.path.exists(yamlPath):
        update_config(yamlPath)
    else:
        gen_config(yamlPath)
    test_log_path = os.path.splitext(s_args.model)[0] + ('_protocol#1.log' if not config.TEST.isPA else '_protocol#2.log')
    print(test_log_path)

    logger = LOG(test_log_path, config.DEBUG)
    logger.info('Training config:{}\n'.format(pprint.pformat(config)))
    logger.info('Using Model',s_args.model)

    # define context
    if config.useGPU:
        ctx = [mx.gpu(int(i)) for i in config.gpu.split(',')]
    else:
        ctx = mx.cpu()
    logger.info("Using context:", ctx)

    # network
    net = get_net(config)
    net.collect_params().load(s_args.model, ctx=ctx)

    results = list()
    for act in HM_act_idx:
        # dataset
        test_imdbs = []
        for i in range(len(config.DATASET.test_image_set)):
            logger.info("Construct Dataset:", config.DATASET.dbname[i], "  Dataset Path:", config.DATASET.dataset_path[i])
            test_imdbs.append(eval(config.DATASET.dbname[i])(config.DATASET.test_image_set[i],
                                                            config.DATASET.root_path[i],
                                                            config.DATASET.dataset_path[i],
                                                            config.final_Model_path))
        data_names  = ['hm36data']
        label_names = ['hm36label']
        test_data_iter = JointsDataIter(test_imdbs[0], runmode=2,
                                        data_names = data_names, label_names=label_names,
                                        shuffle=False, batch_size=len(ctx)*config.TEST.batchsize, logger=logger, action='%02d'%act)

        # define loss and metric
        mean3d = test_data_iter.get_meanstd()['mean3d']
        std3d  = test_data_iter.get_meanstd()['std3d']
        loss = MeanSquareLoss()
        action_metric = MPJPEMetric('Action%02d_metric'%act, mean3d, std3d, pa = config.TEST.isPA)
        xyz_metric    = XYZMetric('XYZ_Action%02d_metric'%act, mean3d, std3d, pa = config.TEST.isPA)

        if not isinstance(test_data_iter, mx.io.PrefetchingIter):
            test_data_iter = mx.io.PrefetchingIter(test_data_iter)

        # train and valid
        act_result = TestNet(net, test_data_iter, loss, action_metric, xyz_metric, mean3d, std3d, config, logger, ctx)
        results.append(act_result)

    LogResult(logger, config, results)
    logger.kill()


if __name__ == '__main__':
    main()