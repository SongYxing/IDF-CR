import os
import shutil


def copydir(inputdir, savedir, mask):
    imgnames = os.listdir(inputdir)

    for imgname in imgnames:
        imgpath = os.path.join(inputdir, imgname)
        copyimgpath = os.path.join(savedir, imgname[19:])
        shutil.copy(imgpath, copyimgpath)


def div(inputpath, savepath, attribute):
    testlist = ['1', '15', '44', '75', '100', '117', '126', '140', '147']
    dirs = os.listdir(inputpath)
    for dir in dirs:
        inputDir = os.path.join(inputPath, dir)
        print(inputDir)
        _, tmp = dir.split('_')
        # if testlist.count(tmp) == 0:
        #     trainpath = os.path.join(savepath, 'train', attribute)
        #     copydir(inputDir, trainpath, tmp)
        # else:
        #     testpath = os.path.join(savepath, 'test', attribute)
        #     copydir(inputDir, testpath, tmp)

        trainpath = os.path.join(savepath, 'train', attribute)
        copydir(inputDir, trainpath, tmp)


inputPath = ''
savePath = ''
div(inputPath, savePath, 'mask')  # mask19,label19,cloudy26
