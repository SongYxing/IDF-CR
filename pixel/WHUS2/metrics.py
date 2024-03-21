import os
import cv2
from image_similarity_measures.evaluate import evaluation
from metrics2 import psnr,ssim

referpath = '/media/abc/Disk14T/2022/SYX/tgrs/dataset/WHhS2_visual/clearDNclips'
outputpath = '/media/abc/Disk14T/2022/SYX/tgrs/result/latent'

bands = os.listdir(outputpath)

for band in bands:
    sum_psnr = 0.0
    sum_ssim = 0.0
    i = 0
    imgnames = os.listdir(os.path.join(outputpath, band))
    for imgname in imgnames:
        refpath = os.path.join(os.path.join(referpath, band), imgname)
        predpath = os.path.join(os.path.join(outputpath, band), imgname)
        pred = cv2.imread(predpath)
        clear = cv2.imread(refpath)
        psnr_result = psnr(pred, clear)
        ssin_result = ssim(pred, clear)
        #result = evaluation(refpath, predpath, ["psnr", "ssim"])
        sum_psnr += psnr_result
        sum_ssim += ssim_result
        i = i + 1
    avg_psnr = sum_psnr / len(imgnames)
    avg_ssim = sum_ssim / len(imgnames)
    print("band:", band, "  psnr:", avg_psnr, "  ssim:", avg_ssim)
