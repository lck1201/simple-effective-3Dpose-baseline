import os
import time
import random
import pickle
from random import shuffle
import multiprocessing

from concurrent import futures
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
from config import config as cfg

from dataset.hm36 import *

ex = futures.ThreadPoolExecutor(multiprocessing.cpu_count())

def get_batch_samples(dbs):
    num_images = len(dbs)
    data = []
    label = []
    folder = []

    if cfg.DEBUG:
        for sample in dbs:
            targets = get_sample(sample)
            data.append(targets[0])
            label.append(targets[1])
            folder.append(targets[2])
    else:
        begin = time.time()
        args = [{
                    'sample': sample
                } for sample in dbs]

        targets = ex.map(get_sample_worker, args)

        for target in targets:
            data.append(target[0])
            label.append(target[1])
            folder.append(target[2])

        end = time.time()
        
    return {
            'hm36data': data,
            'hm36label': label,
            'hm36folder': folder
            }

def get_sample_worker(args):
    return get_sample(args['sample'])

def get_sample(sample):
    joints2d = sample['joints_2d'].copy()
    joints3d = sample['joints_3d'].copy()
    folder   = sample['folder']

    data  = joints2d.copy()
    label = joints3d.copy()

    return data, label, folder

def normalizeDB(rawDB, meanstd):
    mean2d = meanstd['mean2d']
    std2d  = meanstd['std2d']
    mean3d = meanstd['mean3d']
    std3d  = meanstd['std3d']

    for i in range(len(rawDB)):
        rawDB[i]['joints_2d'] = (rawDB[i]['joints_2d'].flatten() - mean2d)/ std2d
        rawDB[i]['joints_3d'] = (rawDB[i]['joints_3d'].flatten() - mean3d)/ std3d

    return rawDB

class JointsDataIter(mx.io.DataIter):
    def __init__(self, db, runmode, data_names, label_names, shuffle, batch_size, logger, action=None):
        super(JointsDataIter, self).__init__()
        self.runmode = runmode

        rawDB = None
        if self.runmode == 0 or self.runmode == 1:
            rawDB = db.gt_db(logger)
        else:
            rawDB = db.gt_db_actions(action, logger)

        self.mean2d, self.std2d, self.mean3d, self.std3d = db.get_meanstd(rawDB, logger)
        self.db = normalizeDB(rawDB, {'mean2d':self.mean2d, 'std2d':self.std2d, 'mean3d':self.mean3d, 'std3d':self.std3d})

        self.size = len(self.db)
        self.index = np.arange(self.size)
        self.shuffle = shuffle
        self.joint_num = db.joint_num
        # self.cfg = cfg
        self.batch_size = batch_size

        self.cur = 0
        self.batch = None

        self.data_names = data_names
        self.label_names = label_names

        # status variable for synchronization between get_data and get_label
        self.data  = None
        self.label = None


        # get first batch to fill in provide_data and provide_label
        self.reset()          # Reset and shuffle
        self.get_batch()

    def get_meanstd(self):
        return {'mean2d':self.mean2d, 'std2d':self.std2d, 'mean3d':self.mean3d, 'std3d':self.std3d}

    def get_size(self):
        return self.size

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_names, self.label)] if self.label_names else None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            data_batch = mx.io.DataBatch(data=self.data, label=self.label,
                                         pad=self.getpad(), index=self.getindex(),
                                         provide_data=self.provide_data, provide_label=self.provide_label)
            return data_batch
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur                                                     # start index
        cur_to = min(cur_from + self.batch_size, self.size)                     # end index
        joints_db = [self.db[self.index[i]] for i in range(cur_from, cur_to)]   # fetch the data

        rst = get_batch_samples(joints_db)
        self.data  = [mx.nd.array(rst[key]) for key in self.data_names]
        self.label = [mx.nd.array(rst['hm36label']), mx.nd.array(rst['hm36folder'])]