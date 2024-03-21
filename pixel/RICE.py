import shutil
import os

txtPath = '/home/abc/SongFlies/GRSL/SpA-GAN_for_cloud_removal-master/data/RICE_DATASET/RICE2/test_list.txt'
oriPath = '/home/abc/SongFlies/GRSL/SpA-GAN_for_cloud_removal-master/data/RICE_DATASET/RICE2/cloudy_image'
savePath = '/home/abc/SongFlies/GRSL/dataset/RICE_DATASET_div/RICE2/test/cloudy_image'


with open(txtPath,'r') as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]

for line in lines:
    oriImagePath = os.path.join(oriPath, line)
    saveImagePath = os.path.join(savePath, line)
    shutil.copy(oriImagePath,saveImagePath)
