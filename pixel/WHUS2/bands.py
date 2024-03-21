import os
import cv2
from gdaldiy import *
from allnet import *
import numpy as np


def to13bands(iuputpath, outpath, folder, band):
    img = imgread(iuputpath)
    h, w, c = img.shape
    img = img / 10000.0
    img = liner_2(img) * 255.0
    path_folder = os.path.join(outpath, folder)

    for i in range(c):
        if not os.path.exists(os.path.join(path_folder, band + str(i))):
            os.mkdir(os.path.join(path_folder, band + str(i)))
        srcPath = iuputpath.split('/')
        finalpath = os.path.join(os.path.join(path_folder, band + str(i)), srcPath[-2] + '_' + srcPath[-1].split('.')[-2] + '.png')
        cv2.imwrite(finalpath, img[:, :, i])


srcpath = ''
savepath = ''

folders = os.listdir(srcpath)
for folder in folders:
    bands = os.listdir(os.path.join(srcpath, folder))
    if not os.path.exists(os.path.join(savepath, folder)):
        os.mkdir(os.path.join(savepath, folder))
    else:
        exit()
    for band in bands:
        print(band)
        imagepaths = make_test_data_list(os.path.join(os.path.join(srcpath, folder), band))
        for imagepath in imagepaths:
            to13bands(imagepath, savepath,folder, band)
