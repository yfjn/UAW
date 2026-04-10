import sys
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/PycharmProjects/UAW")
import time
from PIL import Image, ImageFile
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import yaml
import numpy as np
from easydict import EasyDict
from Tools.ImageIO import logger_config
from dataset import ImageDataset
from UAWUP.uaw_attack import RepeatAdvPatch_Attack
from UDUP.Auxiliary import *
from AllConfig.GConfig import abspath
from model import WatermarkExtractor, SimpleWatermarkExtractor
# import warnings
# warnings.filterwarnings("ignore")


if __name__ == '__main__':
    bit_num = 64
    batch_size = 1
    epochs = 1000
    min_e, max_e = 0, 600
    lr = 1e-4
    extractor_mode = 'convnext_small'
    resume_path = ''   #'results/results_extractor/size=30_step=3_eps=120_lambdaw=0.1/complex_extractor_999_100.00.pth'
    transform = torchvision.transforms.ToTensor()
    data_root = os.path.join(abspath, "AllData")
    train_dataset = ImageDataset(transform=transform, length=100, test=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # val_dataset = ImageDataset(transform=transform, length=10, test=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    device = torch.device("cuda")
    patch1 = torch.load("results/results_uawup/size=30_step=3_eps=120_lambdaw=0.1/advtorch/advpatch1_30", map_location='cpu').detach().to(device)
    patch2 = torch.load("results/results_uawup/size=30_step=3_eps=120_lambdaw=0.1/advtorch/advpatch2_30", map_location='cpu').detach().to(device)
    # patch1 = torch.load("results/result_uawup/size=30_step=3_eps=120_lambdaw=0.1/advtorch/advpatch1_27", map_location='cpu').detach().to(device)
    # patch2 = torch.load("results/result_uawup/size=30_step=3_eps=120_lambdaw=0.1/advtorch/advpatch2_27", map_location='cpu').detach().to(device)
    patch1.requires_grad = True
    patch2.requires_grad = True

    # if extractor_mode == 'complex':
        # extractor = WatermarkExtractor(bit_num=bit_num).to(device)
    extractor = torchvision.models.convnext_small(weights=None)
    extractor.classifier[2] = nn.Linear(768, bit_num)
    extractor.classifier.add_module("3", nn.Sigmoid())
    extractor.to(device)
    # elif extractor_mode == 'simple':
    #     extractor = SimpleWatermarkExtractor(bit_num=bit_num).to(device)
    optimizer = torch.optim.AdamW(extractor.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=5) # 让模型周期性地调整学习率，避免陷入局部最优
    BCE = nn.BCEWithLogitsLoss().cuda()  # 模型的原始输出（未经过sigmoid）
    # BCE = nn.BCELoss().cuda()  # 会进行sigmoid
    save_dir = os.path.join(abspath, 'results/results_extractor/size=30_step=3_eps=120_lambdaw=0.1_new')
    # save_dir = os.path.join(abspath, 'results/result_extractor/size=30_step=3_eps=120_lambdaw=0.1')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = logger_config(log_filename='results/Mylog_extractor/size=30_step=3_eps=120_lambdaw=0.1_new.log')
    # logger = logger_config(log_filename='results/mylog_extractor/size=30_step=3_eps=120_lambdaw=0.1.log')
    logger.info(f'bit_num={bit_num}, batch_size={batch_size}, epochs={min_e}-{max_e}-{epochs}, lr={lr}, extractor_mode={extractor_mode}')

    if resume_path:
        extractor.load_state_dict(torch.load(resume_path, map_location='cpu'), strict=True)
        filename = resume_path.split('/')[-1].split('.pth')[0]
        s = int(filename.split('_')[2])
        # s = 0
        best_bit_acc = float(filename.split('_')[3])
        logger.info("resume training, epoch: {}, bit_acc:{:.2f}".format(s, best_bit_acc))
    else:
        # logger.info(extractor)
        s = 0
        best_bit_acc = 50.
        logger.info("start training=====================")
    elapsed_time = 0
    for e in range(s, epochs):
        start_time = time.time()
        extractor.train()
        total_bce_loss = 0.
        total_bit_acc = 0.
        for k, (textimg, bits) in enumerate(train_dataloader):
            textimg = textimg.to(device)
            bits = bits.to(device)
            
            patch_h, patch_w = patch1.shape[2:]
            h_real, w_real = textimg.shape[2:]
            wm_block = torch.ones((batch_size, 3, 240, 240)).cuda()
            for r in range(8):
                for c in range(8):
                    if bits[0, r*8 + c] == 0:
                        wm_block[:, :, r * patch_h: r * patch_h + patch_h, c * patch_w: c * patch_w + patch_w] = patch1
                    else:
                        wm_block[:, :, r * patch_h: r * patch_h + patch_h, c * patch_w: c * patch_w + patch_w] = patch2
            h_num = h_real // patch_h // 8 + 1
            w_num = w_real // patch_w // 8 + 1
            AWU = wm_block.repeat(1, 1, h_num, w_num)
            AWU = AWU[:, :, :h_real, :w_real]
            # m = torch.where((background_image - textimg) == 0, 1, 0)  # 这样错误，一种颜色255也会被认为是背景
            m = extract_background(textimg)
            mixup_weight = min(1, max(0, (e-min_e) / (max_e-min_e)))
            AWI = AWU * m + (textimg*mixup_weight+AWU*(1-mixup_weight)) * (1-m)
            AWI = torch.clamp(AWI, min=0, max=1)
            save_adv_patch_img(AWU, os.path.join(save_dir, "awu.png"))
            save_adv_patch_img(AWI, os.path.join(save_dir, "awi.png"))

            wm_blocks = []
            for r in range(4):
                for c in range(4):
                    wm_block = AWI[:, :, r*240:(r+1)*240, c*240:(c+1)*240]
                    wm_blocks.append(wm_block)
            save_adv_patch_img(wm_blocks[0], os.path.join(save_dir, "wm_block.png"))
            # wm_blocks = torch.stack(wm_blocks)  # 错误，会生成(16, batch_size, 3, 240, 240))
            wm_blocks = torch.cat(wm_blocks, dim=0)  # (16*batch_size, 3, 240, 240))
            # wm_blocks = wm_blocks.view(4, 4, 3, 240, 240)  # (rows, cols, C, H, W)
            # wm_blocks = wm_blocks.permute(2, 0, 3, 1, 4)  # (C, rows, H, cols, W)
            # wm_blocks = wm_blocks.reshape(3, 960, 960)  # (C, 960, 960)
            # save_adv_patch_img(wm_blocks.unsqueeze(0), os.path.join(save_dir, "wm_blocks.png"))
            bits_hat = extractor(wm_blocks)
            bce_loss = BCE(bits_hat, bits.repeat(16, 1))
            total_bce_loss += bce_loss
            bit_acc = get_bit_acc(bits_hat, bits.repeat(16, 1))
            total_bit_acc += bit_acc
            sys.stdout.write('\r')
            sys.stdout.write("epoch: {}, bce_loss:{:.6f}, bit_acc:{:.6f}".format(e, total_bce_loss/(k+1), total_bit_acc/(k+1)))
            sys.stdout.flush()

            optimizer.zero_grad()
            bce_loss.backward()
            optimizer.step()
            # scheduler.step()
            torch.cuda.empty_cache()

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        sys.stdout.write('\n')
        logger.info("epoch: %d, bce_loss: %.6f, bit_acc:%.6f, elapsed_time: %d:%02d:%02d" % (e, total_bce_loss/100, total_bit_acc/100, *get_hms(elapsed_time)))
        if total_bit_acc > best_bit_acc or total_bit_acc > 99.99:
            best_bit_acc = total_bit_acc
            torch.save(extractor.state_dict(), os.path.join(save_dir, '{}_extractor_{}_{:.2f}.pth'.format(extractor_mode, e, total_bit_acc)))
