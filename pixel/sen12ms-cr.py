import tifffile
from PIL import Image
import os
import numpy as np


def tiff2rgb_13(inputpath, savepath):
    img = tifffile.imread(inputpath)
    img = img[:, :, 1:4]
    n_max = np.max(img)
    n_min = np.min(img)
    img = ((img - n_min) / (n_max - n_min)) * 255.0
    img = np.uint8(img)
    img = img[..., ::-1]
    img = Image.fromarray(img)
    img.save(savepath)


def tiff2rgb_2_1(inputpath, savepath):
    img = tifffile.imread(inputpath)
    img = 0.5 * img[:, :, 0] + 0.5 * img[:, :, 1]
    n_max = np.max(img)
    n_min = np.min(img)
    img = ((img - n_min) / (n_max - n_min)) * 255.0
    img = np.uint8(img)
    img = Image.fromarray(img)
    img.save(savepath)


def tiff2rgb_2_2(inputpath, savepath):
    img = tifffile.imread(inputpath)
    n_max = np.max(img)
    n_min = np.min(img)
    img = ((img - n_min) / (n_max - n_min)) * 255.0
    img = np.uint8(img)
    img = Image.fromarray(img[:,:,0])
    img.save(savepath)


tiffpath = '/media/abc/DA18EBFA09C1B27D/Song/dataset/SEN12MS-CR-TS/ROIs1158_spring_s1'
rgbpath = '/media/abc/DA18EBFA09C1B27D/Song/dataset/SEN12MS-CR-TS/ROIs1158_spring_s1_rgb_2'

dirs = os.listdir(tiffpath)
for dir in dirs:
    rgbdir = os.path.join(rgbpath, dir)
    if not os.path.exists(rgbdir):
        os.mkdir(rgbdir)
    tiffdir = os.path.join(tiffpath, dir)
    tiffnames = os.listdir(tiffdir)
    for tiffname in tiffnames:
        tiff = os.path.join(tiffdir, tiffname)
        imgname, _ = os.path.splitext(tiffname)
        rgb = os.path.join(rgbdir, imgname + '.png')
        tiff2rgb_2_2(tiff, rgb)

# dir = '/media/abc/DA18EBFA09C1B27D/Song/dataset/sen12ms_test/concat3/ROIs1158_spring_s2_cloudy'
# dst = '/home/abc/SongFlies/GRSL/dataset/sen12ms-cr_div/spa-swin-test/test/cloudy'
# imgnames = os.listdir(dir)
# for imgname in imgnames:
#     srcpath = os.path.join(dir, imgname)
#     pngname, _ = os.path.splitext(imgname)
#     dstpath = os.path.join(dst,  pngname + '.png')
#     tiff2rgb_13(srcpath, dstpath)
