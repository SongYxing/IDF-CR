import os
import random
import shutil
import yaml
from attrdict import AttrMap
import time

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

from data_manager_whus2 import TrainDataset
from models.gen.SPANet_whus2 import Generator
from models.dis.dis import Discriminator
import utils
from utils import gpu_manage, save_image, checkpoint_gen
from eval_whus2 import test
from log_report import LogReport
from log_report import TestReport


def train(config):
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:{}".format(config.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:{}".format(config.gpu_ids[0]))
    gpu_manage(config)
    ### DATASET LOAD ###
    print('===> Loading datasets')
    dataset = TrainDataset(config)
    print('dataset:', len(dataset))
    train_size = int((1 - config.validation_size) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    print('train dataset:', len(train_dataset))
    print('validation dataset:', len(validation_dataset))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads,
                                        batch_size=config.validation_batchsize, shuffle=False)

    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    # dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)
    #
    # if config.dis_init is not None:
    #     param = torch.load(config.dis_init)
    #     dis.load_state_dict(param)
    #     print('load {} as pretrained model'.format(config.dis_init))

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    # opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    # real_a = torch.FloatTensor(config.batchsize, config.in_ch, config.width, config.height)
    # real_b = torch.FloatTensor(config.batchsize, config.out_ch, config.width, config.height)
    # real_sar = torch.FloatTensor(config.batchsize, 1, config.width, config.height)
    # M = torch.FloatTensor(config.batchsize, config.width, config.height)

    real_x10 = torch.FloatTensor(config.batchsize, config.in_ch_10, config.width_10, config.height_10)
    real_x20 = torch.FloatTensor(config.batchsize, config.in_ch_20, config.width_20, config.height_20)
    real_x60 = torch.FloatTensor(config.batchsize, config.in_ch_60, config.width_60, config.height_60)
    real_y10 = torch.FloatTensor(config.batchsize, config.in_ch_10, config.width_10, config.height_10)
    real_y20 = torch.FloatTensor(config.batchsize, config.in_ch_20, config.width_20, config.height_20)
    real_y60 = torch.FloatTensor(config.batchsize, config.in_ch_60, config.width_60, config.height_60)
    M_10 = torch.FloatTensor(config.batchsize, config.width_10, config.height_10)
    M_20 = torch.FloatTensor(config.batchsize, config.width_20, config.height_20)
    M_60 = torch.FloatTensor(config.batchsize, config.width_60, config.height_60)

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()
    if config.cuda:
        gen = nn.DataParallel(gen, device_ids=[0, 1,2,3])
        gen = gen.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        real_x10 = real_x10.cuda()
        real_x20 = real_x20.cuda()
        real_x60 = real_x60.cuda()
        real_y10 = real_y10.cuda()
        real_y20 = real_y20.cuda()
        real_y60 = real_y60.cuda()
        M_10 = M_10.cuda()
        M_20 = M_20.cuda()
        M_60 = M_60.cuda()

    real_x10 = Variable(real_x10)
    real_x20 = Variable(real_x20)
    real_x60 = Variable(real_x60)
    real_y10 = Variable(real_y10)
    real_y20 = Variable(real_y20)
    real_y60 = Variable(real_y60)
    M_10 = Variable(M_10)
    M_20 = Variable(M_20)
    M_60 = Variable(M_60)

    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    print('===> begin')
    start_time = time.time()
    # main
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            real_x10, real_x20, real_x60, real_y10, real_y20, real_y60 = batch[0], batch[1], batch[2], \
                                                                         batch[3], batch[4], batch[5]

            real_x10 = real_x10.cuda()
            real_x20 = real_x20.cuda()
            real_x60 = real_x60.cuda()
            real_y10 = real_y10.cuda()
            real_y20 = real_y20.cuda()
            real_y60 = real_y60.cuda()
            # M_10 = M_10.cuda()
            # M_20 = M_20.cuda()
            # M_60 = M_60.cuda()
            with torch.no_grad():
                real_x10.resize_(real_x10.size()).copy_(real_x10)
                real_x20.resize_(real_x20.size()).copy_(real_x20)
                real_x60.resize_(real_x60.size()).copy_(real_x60)
                real_y10.resize_(real_y10.size()).copy_(real_y10)
                real_y20.resize_(real_y20.size()).copy_(real_y20)
                real_y60.resize_(real_y60.size()).copy_(real_y60)
                # M_10.resize_(M_10.size()).copy_(M_10)
                # M_20.resize_(M_20.size()).copy_(M_20)
                # M_60.resize_(M_60.size()).copy_(M_60)

            fake_y10, fake_y20, fake_y60 = gen.forward(real_x10, real_x20, real_x60)
            opt_gen.zero_grad()

            loss_g_l1_10 = criterionL1(fake_y10, real_y10) * config.lamb
            loss_g_l1_20 = criterionL1(fake_y20, real_y20) * config.lamb
            loss_g_l1_60 = criterionL1(fake_y60, real_y60) * config.lamb

            # loss_g_att_10 = criterionMSE(att10[:, 0, :, :], M_10)
            # loss_g_att_20 = criterionMSE(att20[:, 0, :, :], M_20)
            # loss_g_att_60 = criterionMSE(att60[:, 0, :, :], M_60)
            loss_g = loss_g_l1_10 + loss_g_l1_20 + loss_g_l1_60  # + loss_g_att_60

            loss_g.backward()

            opt_gen.step()

            # log
            if iteration % 10 == 0:
                print(
                    "===> Epoch[{}]({}/{}): loss_d_fake: {:.4f} loss_d_60: {:.4f} loss_g_20: {:.4f} loss_g_10: {:.4f}".format(
                        epoch, iteration, len(training_data_loader), 0, loss_g_l1_60.item(), loss_g_l1_20.item(),
                        loss_g_l1_10.item()))

                log = {}
                log['epoch'] = epoch
                log['iteration'] = len(training_data_loader) * (epoch - 1) + iteration
                log['gen/loss'] = loss_g.item()
                # log['dis/loss'] = loss_d.item()

                logreport(log)

        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        if epoch % 5 == 0:
            with torch.no_grad():
                test(config, validation_data_loader, gen, criterionMSE, epoch)
            print('validation finished')
        if epoch % config.snapshot_interval == 0:
            checkpoint_gen(config, epoch, gen)

        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    print('training time:', time.time() - start_time)


if __name__ == '__main__':
    with open('config_whus2.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    utils.make_manager()
    n_job = utils.job_increment()
    config.out_dir = os.path.join(config.out_dir, '{:06}'.format(n_job))
    os.makedirs(config.out_dir)
    print('Job number: {:04d}'.format(n_job))

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    train(config)
