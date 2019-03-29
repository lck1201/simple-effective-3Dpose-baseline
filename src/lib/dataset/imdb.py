"""
General image database
An image database creates a list of relative image path called image_set_index and
transform index to absolute image path. As to training, it is necessary that ground
truth and proposals are mixed together for training.
roidb
basic format [image_index]
['image', 'height', 'width', 'flipped',
'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import os
import numpy as np
#from PIL import Image
from multiprocessing import Pool, cpu_count

class IMDB(object):
    def __init__(self, name, image_set, root_path, dataset_path, result_path=None):
        """
        basic information about an image database
        :param name: name of image database will be used for any output
        :param root_path: root path store cache
        :param dataset_path: dataset path store images and image lists and label
        """
        self.name = name + '_' + image_set
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path
        self.result_path = result_path

        # abstract attributes
        self.image_set_index = []
        self.num_images = 0

        self.config = {}

    # def image_path_from_index(self, index):
    #     raise NotImplementedError

    # def gt_db(self):
    #     raise NotImplementedError

    # def evaluate_PCKh(self, gt, pred, threshold=0.5):
    #     raise NotImplementedError

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_path = os.path.join(self.data_path, 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    # @property
    # def result_path(self):
    #     if self._result_path and os.path.exists(self._result_path):
    #         return self._result_path
    #     else:
    #         return self.cache_path

    # def image_path_at(self, index):
    #     """
    #     access image at index in image database
    #     :param index: image index in image database
    #     :return: image path
    #     """
        # return self.image_path_from_index(self.image_set_index[index])
