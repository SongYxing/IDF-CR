
import glob
import cv2
import random
import numpy as np
import pickle
import os
from torch.utils import data


class TrainDataset(data.Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # train_list_file = os.path.join(config.datasets_dir, config.train_list)
        # # 如果数据集尚未分割，则进行训练集和测试集的分割
        # if not os.path.exists(train_list_file) or os.path.getsize(train_list_file) == 0:
        #     files = os.listdir(os.path.join(config.datasets_dir, 'ground_truth'))
        #     random.shuffle(files)
        #     n_train = int(config.train_size * len(files))
        #     train_list = files[:n_train]
        #     test_list = files[n_train:]
        #     np.savetxt(os.path.join(config.datasets_dir, config.train_list), np.array(train_list), fmt='%s')
        #     np.savetxt(os.path.join(config.datasets_dir, config.test_list), np.array(test_list), fmt='%s')
        labellist = os.listdir(os.path.join(config.datasets_dir, 'label'))
        cloudylist = os.listdir(os.path.join(config.datasets_dir, 'cloudy'))
        print(cloudylist == labellist)
        self.imlist = os.listdir(os.path.join(config.datasets_dir, 'label'))
        #self.imlist = random.sample(self.imlist, len(self.imlist) // 100)

    def __getitem__(self, index):
        t = cv2.imread(os.path.join(self.config.datasets_dir, 'label', str(self.imlist[index])), 1).astype(
            np.float32)
        x = cv2.imread(os.path.join(self.config.datasets_dir, 'cloudy', str(self.imlist[index])), 1).astype(
            np.float32)

        M = np.clip((t - x).sum(axis=2), 0, 1).astype(np.float32)
        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)

        return x, t, M

    def __len__(self):
        return len(self.imlist)


class TestDataset(data.Dataset):
    def __init__(self, test_dir, in_ch, out_ch):
        super().__init__()
        self.test_dir = test_dir
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.test_files = os.listdir(os.path.join(test_dir, 'cloudy'))

    def __getitem__(self, index):
        filename = os.path.basename(self.test_files[index])

        x = cv2.imread(os.path.join(self.test_dir, 'cloudy', filename), 1).astype(np.float32)

        x = x / 255

        x = x.transpose(2, 0, 1)

        return x, filename

    def __len__(self):
        return len(self.test_files)
