import tifffile
from PIL import Image
import os
import numpy as np


def arr2npy(inputpath, savepath):
    img = tifffile.imread(inputpath)
    np.save("../data/arr.npy", arr)




tiffpath = '/home/abc/SongFlies/GRSL/dataset/sen12ms-cr_div/uncr/ROIs1158/1/S2_cloudy'
rgbpath = '/home/abc/SongFlies/GRSL/dataset/sen12ms-cr_div/uncr'

imagenames = os.listdir(tiffpath)
for imagename in imagenames:
    img = tifffile.imread(os.path.join(tiffpath,imagename))


