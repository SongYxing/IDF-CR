# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:07:54 2018

@author: Neoooli
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

from collections import OrderedDict

from WHUS2.gdaldiy import *
import glob
import os


def randomflip(input_, n):
    # 生成-3到2的随机整数，-1顺时针90度，-2顺时针180，-3顺时针270,0垂直翻转，1水平翻转，2不变
    if n < 0:
        return np.rot90(input_, n)
    elif -1 < n < 2:
        return np.flip(input_, n)
    else:
        return input_


def read_img(datapath, scale=255):
    img = imgread(datapath)
    img[img > scale] = scale
    img = img / scale
    return img


def read_img_woscale(datapath):
    img = imgread(datapath)
    return img


def read_imgs(datapath, scale=255, k=2):
    img_list = []
    l = len(datapath)
    for i in range(l):
        img = read_img(datapath[i], scale)
        img = randomflip(img, k)
        img = img[np.newaxis, :]
        img_list.append(img)
    imgs = np.concatenate(img_list, axis=0)
    return imgs


# def iterate_img(file_list, batch_size=1, rn_list=None, scale=10000, num__cores=tf.data.experimental.AUTOTUNE):
#     if np.any(rn_list) == None:
#         rn_list = [2 for _ in range(len(file_list))]  # 如果不指定翻转参数，就不翻转
#
#     def gen(file_name, k):
#         # print('load img')
#         img = read_img(file_name.numpy().decode('utf-8'), scale)
#         img = randomflip(img, k)
#         return img
#
#     def iterator():
#         dataset = tf.data.Dataset.from_tensor_slices((file_list, rn_list))
#         dataset = dataset.map(lambda x, y: tf.py_function(gen, [x, y], tf.float32), num_parallel_calls=num__cores)
#         dataset = dataset.batch(batch_size)
#         dataset = dataset.prefetch(num__cores)
#         return dataset
#
#     return iterator()

def liner_2(input_):  # 2%线性拉伸,返回0~1之间的值
    def strech(img):
        low, high = np.percentile(img, (2, 98))
        img[low > img] = low
        img[img > high] = high
        return (img - low) / (high - low + 1e-10)

    if len(input_.shape) > 2:
        for i in range(input_.shape[-1]):
            input_[:, :, i] = strech(input_[:, :, i])
    else:
        input_ = strech(input_)
    return input_


def make_test_data_list(data_path):  # make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    filepath = glob.glob(os.path.join(data_path, "*"))  # 读取全部的x域图像路径名称列表
    image_path_lists = []
    for i in range(len(filepath)):
        path = glob.glob(os.path.join(filepath[i], "*"))
        for j in range(len(path)):
            image_path_lists.append(path[j])
    return image_path_lists


def get_write_picture(row_list):  # get_write_picture函数得到训练过程中的可视化结果
    row_ = []
    for i in range(len(row_list)):
        row = row_list[i]
        col_ = []
        for image in row:
            x_image = image[:, :, [2, 1, 0]]
            if i < 1:
                x_image = liner_2(x_image)
            col_.append(x_image)
        row_.append(np.concatenate(col_, axis=1))
    if len(row_list) == 1:
        output = np.concatenate(col_, axis=1)
    else:
        output = np.concatenate(row_, axis=0)  # 得到训练中可视化结果
    return output * 255
