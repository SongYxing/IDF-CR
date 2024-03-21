import glob
import cv2
import random
import numpy as np
import pickle
import os
from torch.utils import data
from WHUS2.allnet import *
from WHUS2.gdaldiy import *


class TrainDataset(data.Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config

        labelPath = os.path.join(config.datasets_dir, 'clearDNclips')
        cloudyPath = os.path.join(config.datasets_dir, 'cloudDNclips')
        self.cloudylist_10 = make_test_data_list(os.path.join(cloudyPath, '10m'))
        self.cloudylist_20 = make_test_data_list(os.path.join(cloudyPath, '20m'))
        self.cloudylist_60 = make_test_data_list(os.path.join(cloudyPath, '60m'))

        self.labellist_10 = make_test_data_list(os.path.join(labelPath, '10m'))
        self.labellist_20 = make_test_data_list(os.path.join(labelPath, '20m'))
        self.labellist_60 = make_test_data_list(os.path.join(labelPath, '60m'))

        # print(cloudylist == labellist)
        # self.imlist = os.listdir(os.path.join(config.datasets_dir, 'label'))
        # random.seed(42)
        # self.cloudylist_10 = random.sample(self.cloudylist_10, len(self.cloudylist_10) // 10)
        # random.seed(42)
        # self.cloudylist_20 = random.sample(self.cloudylist_20, len(self.cloudylist_20) // 10)
        # random.seed(42)
        # self.cloudylist_60 = random.sample(self.cloudylist_60, len(self.cloudylist_60) // 10)
        #
        # random.seed(42)
        # self.labellist_10 = random.sample(self.labellist_10, len(self.labellist_10) // 10)
        # random.seed(42)
        # self.labellist_20 = random.sample(self.labellist_20, len(self.labellist_20) // 10)
        # random.seed(42)
        # self.labellist_60 = random.sample(self.labellist_60, len(self.labellist_60) // 10)

    def __getitem__(self, index):
        cloudy10 = read_img_woscale(self.cloudylist_10[index]).astype(
            np.float32)
        cloudy20 = read_img_woscale(self.cloudylist_20[index]).astype(
            np.float32)
        cloudy60 = read_img_woscale(self.cloudylist_60[index]).astype(
            np.float32)
        label10 = read_img_woscale(self.labellist_10[index]).astype(
            np.float32)
        label20 = read_img_woscale(self.labellist_20[index]).astype(
            np.float32)
        label60 = read_img_woscale(self.labellist_60[index]).astype(
            np.float32)

        # M_10 = np.clip((label10 - cloudy10).sum(axis=2), 0, 1).astype(np.float32)
        # M_20 = np.clip((label20 - cloudy20).sum(axis=2), 0, 1).astype(np.float32)
        # M_60 = np.clip((label60 - cloudy60).sum(axis=2), 0, 1).astype(np.float32)

        cloudy10 = cloudy10 / 10000
        cloudy20 = cloudy20 / 10000
        cloudy60 = cloudy60 / 10000

        label10 = label10 / 10000
        label20 = label20 / 10000
        label60 = label60 / 10000

        cloudy10 = cloudy10.transpose(2, 0, 1)
        cloudy20 = cloudy20.transpose(2, 0, 1)
        cloudy60 = cloudy60.transpose(2, 0, 1)

        label10 = label10.transpose(2, 0, 1)
        label20 = label20.transpose(2, 0, 1)
        label60 = label60.transpose(2, 0, 1)

        return cloudy10, cloudy20, cloudy60, label10, label20, label60 #, M_10, M_20, M_60

    def __len__(self):
        return len(self.cloudylist_10)


class TestDataset(data.Dataset):
    def __init__(self, test_dir, in_ch, out_ch):
        super().__init__()
        self.test_dir = test_dir
        self.cloudylist_10 = make_test_data_list(os.path.join(test_dir, '10m'))
        self.cloudylist_20 = make_test_data_list(os.path.join(test_dir, '20m'))
        self.cloudylist_60 = make_test_data_list(os.path.join(test_dir, '60m'))

    def __getitem__(self, index):
        filename = self.cloudylist_10[index]
        cloudy10 = read_img_woscale(self.cloudylist_10[index]).astype(
            np.float32)
        cloudy20 = read_img_woscale(self.cloudylist_20[index]).astype(
            np.float32)
        cloudy60 = read_img_woscale(self.cloudylist_60[index]).astype(
            np.float32)

        cloudy10 = cloudy10 / 10000
        cloudy20 = cloudy20 / 10000
        cloudy60 = cloudy60 / 10000
        cloudy10 = cloudy10.transpose(2, 0, 1)
        cloudy20 = cloudy20.transpose(2, 0, 1)
        cloudy60 = cloudy60.transpose(2, 0, 1)

        return cloudy10, cloudy20, cloudy60, filename

    def __len__(self):
        return len(self.cloudylist_10)
