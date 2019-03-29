import yaml
from easydict import EasyDict as edict
import mxnet

config = edict()

config.MXNET_VERSION = 'mxnet-version' + mxnet.__version__
config.block = 'Martinez Baseline'
config.saveModel_path = './output/model/'
config.final_Model_path = ''
config.train_log_path = './output/train-log/'

config.DEBUG = False
config.useGPU = True
config.gpu = '0'

config.resume = False
config.resumeckp = ''

#network-related config
config.NETWORK = edict()
config.NETWORK.nResBlock = 2
config.NETWORK.nJoints = 16
config.NETWORK.hybrid = True

#train-related config
config.TRAIN = edict()
config.TRAIN.batchsize = 64
config.TRAIN.optimizer = 'adam'
config.TRAIN.lr = 0.001
config.TRAIN.decay_rate = 0.96
config.TRAIN.decay_steps = 100000
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 200
config.TRAIN.SHUFFLE = True
# config.TRAIN.UseMetric = False

# dataset-related config
config.DATASET = edict()
config.DATASET.dbname = ['hm36']
config.DATASET.train_image_set = ['train']
config.DATASET.valid_image_set = ['valid']
config.DATASET.test_image_set  = ['test']
config.DATASET.root_path = ['/home/chuankang/code/simple-effective-3Dpose-baseline/']
config.DATASET.dataset_path = ['/home/chuankang/HardDrive4T/data_chuankang/hm36/']
config.DATASET.sigma = 0

# test-related config
config.TEST = edict()
config.TEST.batchsize = 64
config.TEST.isPA = False

def update_config(config_file):
    # exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])

