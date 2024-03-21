import numpy as np
import torch
from utils import save_image


def test(config, test_data_loader, gen, criterionMSE, epoch):
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:{}".format(config.gpu_ids[0]))
    device = torch.device("cuda:{}".format(config.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    avg_mse10 = 0
    avg_mse20 = 0
    avg_mse60 = 0
    avg_psnr10 = 0
    avg_psnr20 = 0
    avg_psnr60 = 0
    for i, batch in enumerate(test_data_loader):
        real_x10, real_x20, real_x60, real_y10, real_y20, real_y60= batch[0], batch[1], batch[2], \
                                                                                       batch[3], batch[4], batch[5]

        if config.cuda:
            real_x10 = real_x10.to(device)
            real_x20 = real_x20.to(device)
            real_x60 = real_x60.to(device)
            real_y10 = real_y10.to(device)
            real_y20 = real_y20.to(device)
            real_y60 = real_y60.to(device)

        out10, out20, out60 = gen(real_x10, real_x20, real_x60)

        mse10 = criterionMSE(out10, real_y10)
        mse20 = criterionMSE(out20, real_y20)
        mse60 = criterionMSE(out60, real_y60)
        psnr10 = 10 * np.log10(1 / mse10.item())
        psnr20 = 10 * np.log10(1 / mse20.item())
        psnr60 = 10 * np.log10(1 / mse60.item())

        avg_mse10 += mse10.item()
        avg_mse20 += mse20.item()
        avg_mse60 += mse60.item()
        avg_psnr10 += psnr10
        avg_psnr20 += psnr20
        avg_psnr60 += psnr60

    avg_mse10 = avg_mse10 / len(test_data_loader)
    avg_psnr10 = avg_psnr10 / len(test_data_loader)

    avg_mse20 = avg_mse20 / len(test_data_loader)
    avg_psnr20 = avg_psnr20 / len(test_data_loader)
    avg_mse60 = avg_mse60 / len(test_data_loader)
    avg_psnr60 = avg_psnr60 / len(test_data_loader)

    print("===> Avg. MSE10: {:.4f}".format(avg_mse10))
    print("===> Avg. PSNR10: {:.4f} dB".format(avg_psnr10))


    print("===> Avg. MSE20: {:.4f}".format(avg_mse20))
    print("===> Avg. PSNR20: {:.4f} dB".format(avg_psnr20))

    print("===> Avg. MSE60: {:.4f}".format(avg_mse60))
    print("===> Avg. PSNR60: {:.4f} dB".format(avg_psnr60))


