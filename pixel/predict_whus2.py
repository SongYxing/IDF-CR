import numpy as np
import argparse
from tqdm import tqdm
import yaml
from attrdict import AttrMap

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_manager_whus2 import TestDataset
from utils import gpu_manage, save_image, heatmap
from models.gen.SPANet_whus2 import Generator
import os
from WHUS2.gdaldiy import imgwrite


def predict(config, args):
    device = torch.device("cuda:{}".format(config.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    gpu_manage(args)
    dataset = TestDataset(args.test_dir, config.in_ch, config.out_ch)
    data_loader = DataLoader(dataset=dataset, num_workers=config.threads, batch_size=1, shuffle=False)

    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    param = torch.load(args.pretrained)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in param.items():
        name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    # load params

    gen.load_state_dict(new_state_dict)

    if args.cuda:
        gen = gen.cuda()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            cloudy10, cloudy20, cloudy60, filename = batch[0], batch[1], batch[2], batch[3]
            filename = filename[0].split('/')
            savepath = args.out_dir + '/10m/' + filename[-2]
            savepath1 = args.out_dir + '/20m/' + filename[-2]
            savepath2 = args.out_dir + '/60m/' + filename[-2]
            if not os.path.exists(savepath):  # 如果保存x域测试结果的文件夹不存在则创建
                os.makedirs(savepath)
            if not os.path.exists(savepath1):  # 如果保存x域测试结果的文件夹不存在则创建
                os.makedirs(savepath1)
            if not os.path.exists(savepath2):  # 如果保存x域测试结果的文件夹不存在则创建
                os.makedirs(savepath2)
            if config.cuda:
                cloudy10 = cloudy10.to(device)
                cloudy20 = cloudy20.to(device)
                cloudy60 = cloudy60.to(device)

            out10, out20, out60 = gen(cloudy10, cloudy20, cloudy60)
            write_image = out10.cpu().numpy()[0, :, :, :].transpose(1, 2, 0) * 10000
            write_image1 = out20.cpu().numpy()[0, :, :, :].transpose(1, 2, 0) * 10000
            write_image2 = out60.cpu().numpy()[0, :, :, :].transpose(1, 2, 0) * 10000

            savepath = savepath + '/' + filename[-1].split('.')[-2] + '.tif'
            imgwrite(savepath, write_image)
            savepath1 = savepath1 + '/' + filename[-1].split('.')[-2] + '.tif'
            imgwrite(savepath1, write_image1)
            savepath2 = savepath2 + '/' + filename[-1].split('.')[-2] + '.tif'
            imgwrite(savepath2, write_image2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--test_dir', type=str,
                        default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    config = AttrMap(config)

    predict(config, args)

# python predict.py --config config.yml --test_dir /home/abc/SongFlies/GRSL/dataset/RICE/RICE1 --out_dir data/output/try1 --pretrained pretrained_models/RICE1/gen_model_epoch_200.pth --cuda
