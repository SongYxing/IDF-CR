import os
from PIL import Image, ImageFilter
import numpy as np


def camixup(caPath, sarPath, lqPath, savePath, caRatio, cloudyPath):
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    else:
        exit()
    caimgnames = os.listdir(caPath)

    for caimgname in caimgnames:

        # cloudyimg = Image.open(os.path.join(cloudyPath, caimgname))
        # cloudyimg.show()

        caimg = Image.open(os.path.join(caPath, caimgname))
        caimg.convert('L')
        caimg_fliter = caimg.filter(ImageFilter.MedianFilter(5))
        # caimg_fliter.save(
        #     os.path.join('/home/abc/SongFlies/GRSL/dataset/sen12ms-cr_div/for_test/img_fliter', caimgname))

        caimg_fliter_2 = caimg_fliter.point(lambda p: p > 128 and 255)
        # caimg_fliter_2.save(
        #     os.path.join('/home/abc/SongFlies/GRSL/dataset/sen12ms-cr_div/for_test/img_fliter2', caimgname))
        # # caimg_fliter_2.show()

        caimg_fliter_2_np = np.array(caimg_fliter_2)
        caimg_fliter_2_np = caimg_fliter_2_np / 255

        sarimg = Image.open(os.path.join(sarPath, caimgname))
        sarimg = sarimg.convert('RGB')
        # sarimg.show()

        sarimg_np = np.array(sarimg)
        mask = 1 - caimg_fliter_2_np

        lqimg = Image.open(os.path.join(lqPath, caimgname))
        # lqimg.show()
        lqimg_np = np.array(lqimg)
        for i in range(3):
            lqimg_np[:, :, i] = caRatio * mask * lqimg_np[:, :, i] + (1 - caRatio) * mask * sarimg_np[:, :, i] + (
                    1 - mask) * lqimg_np[:, :, i]
        result = Image.fromarray(lqimg_np)
        # result.show()
        result.save(os.path.join(savePath, caimgname))


capath = ''
sarpath = ''
lqpath = ''
savepath = ''
cloudpath = ''
camixup(caPath=capath, sarPath=sarpath, lqPath=lqpath, savePath=savepath, caRatio=0.8, cloudyPath=cloudpath)
