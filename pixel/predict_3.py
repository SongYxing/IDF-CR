import numpy as np
import argparse
from tqdm import tqdm
import yaml
from attrdict import AttrMap

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_manager import TestDataset
from utils import gpu_manage, save_image, heatmap
from models.gen.SPANet import Generator


def predict(config, args):
    device = torch.device("cuda:{}".format(config.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    gpu_manage(args)
    dataset = TestDataset(args.test_dir, config.in_ch, config.out_ch)
    data_loader = DataLoader(dataset=dataset, num_workers=config.threads, batch_size=1, shuffle=False)

    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    param = torch.load(args.pretrained)
    gen.load_state_dict(param)

    if args.cuda:
        gen = gen.cuda()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            x = Variable(batch[0])
            filename = batch[1][0]
            if args.cuda:
                x = x.to(device)

            out = gen(x)

            h = 1
            w = 3
            c = 3
            p = config.width

            allim = np.zeros((h, w, c, p, p))  # (1, 3, 3, 512, 512) 1 3 chw
            x_ = x.cpu().numpy()[0]
            out_ = out.cpu().numpy()[0]
            in_rgb = x_[:3]
            out_rgb = np.clip(out_[:3], 0, 1)
            # att_ = att.cpu().numpy()[0] * 255
            # heat_att = heatmap(att_.astype('uint8'))

            # allim[0, 0, :] = in_rgb * 255 #(3, 512, 512)
            # allim[0, 1, :] = out_rgb * 255
            # allim[0, 2, :] = heat_att
            # allim = allim.transpose(0, 3, 1, 4, 2) # 1 h 3 w c
            # allim = allim.reshape((h*p, w*p, c)) # h 3w c
            # save_image(args.out_dir, allim, i, 1, filename=filename)
            save_image(args.out_dir, out_rgb.transpose(1, 2, 0) * 255, i, 1, filename=filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--test_dir', type=str, default='')
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


