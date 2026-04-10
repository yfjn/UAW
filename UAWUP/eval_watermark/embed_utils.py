import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import random
from Tools.Baseimagetool import *
import nltk
# nltk.download('brown')

def bit_template_mapping(watermark, p_0_t, p_1_t, device=None):
    """
    Module 1: Bit Template Mapping
    根据水印矩阵和两个比特模板生成基础水印块
    全部使用torch.Tensor操作
    
    参数：
    - watermark: 水印矩阵 (wm_h, wm_w) numpy.ndarray或torch.Tensor，值为0或1
    - p_0_t: 比特模板0 torch.Tensor (1, 3, s, s) BGR格式，范围0-1
    - p_1_t: 比特模板1 torch.Tensor (1, 3, s, s) BGR格式，范围0-1
    - device: torch device (None表示CPU)
    
    返回：
    - base_block: 基础水印块 torch.Tensor (1, 3, wm_h*s, wm_w*s) BGR格式，范围0-1
    """
    # 确保在正确的设备上
    if device is not None:
        p_0_t = p_0_t.to(device)
        p_1_t = p_1_t.to(device)
    
    # 转换水印为tensor
    if not torch.is_tensor(watermark):
        watermark_t = torch.from_numpy(watermark).to(device if device else torch.device('cpu'))
    else:
        watermark_t = watermark.to(device if device else torch.device('cpu'))
    
    wm_h, wm_w = watermark_t.shape
    s = p_0_t.shape[-1]
    channels = p_0_t.shape[1]
    
    # 创建基础水印块
    base_block = torch.zeros((1, channels, wm_h * s, wm_w * s), dtype=p_0_t.dtype, device=device if device else torch.device('cpu'))
    
    # 根据水印值填充对应的比特模板
    for i in range(wm_h):
        for j in range(wm_w):
            if watermark_t[i, j] == 0:
                base_block[0, :, i*s:(i+1)*s, j*s:(j+1)*s] = p_0_t
            else:
                base_block[0, :, i*s:(i+1)*s, j*s:(j+1)*s] = p_1_t
    
    return base_block

def flip_based_unit_construction(base_block, wm_h=8, wm_w=8, device=None):
    """
    Module 2: Flip-based Unit Construction
    根据基础水印块生成4个翻转变体（原始、水平翻转、垂直翻转、水平垂直翻转）
    全部使用torch.Tensor操作
    
    参数：
    - base_block: 基础水印块 torch.Tensor (1, 3, wm_h*s, wm_w*s)
    - wm_h: 水印高度
    - wm_w: 水印宽度
    - device: torch device
    
    返回：
    - units: 包含4个单元的列表，每个都是torch.Tensor (1, 3, wm_h*s, wm_w*s)
             [原始, 水平翻转, 垂直翻转, 水平垂直翻转]
    """
    if device is not None:
        base_block = base_block.to(device)
    
    # 生成4个翻转变体
    unit_0 = base_block  # 原始
    unit_1 = torch.flip(base_block, dims=[3])  # 水平翻转
    unit_2 = torch.flip(base_block, dims=[2])  # 垂直翻转
    unit_3 = torch.flip(base_block, dims=[2, 3])  # 水平垂直翻转
    
    return [unit_0, unit_1, unit_2, unit_3]

def tiling_for_underpainting_generation(base_block, num_repeats, device=None):
    """
    Module 3: Tiling for Underpainting Generation
    通过平铺基础水印块并应用翻转生成完整的对抗水印底纹
    全部使用torch.Tensor操作
    
    参数：
    - base_block: 基础水印块 torch.Tensor (1, 3, wm_h*s, wm_w*s)
    - num_repeats: 平铺次数（行和列）
    - device: torch device
    
    返回：
    - underpainting: 对抗水印底纹 torch.Tensor (1, 3, num_repeats*wm_h*s, num_repeats*wm_w*s) BGR格式
    """
    if device is not None:
        base_block = base_block.to(device)
    
    _, channels, block_h, block_w = base_block.shape
    
    # 创建完整底纹图像
    underpainting = torch.zeros((1, channels, num_repeats * block_h, num_repeats * block_w), 
                                dtype=base_block.dtype, device=device if device else torch.device('cpu'))
    
    # 平铺并应用翻转
    for i in range(num_repeats):
        for j in range(num_repeats):
            block = base_block.clone()
            # 根据位置(i,j)应用翻转
            if i % 2 == 1:  # 奇数行，垂直翻转
                block = torch.flip(block, dims=[2])
            if j % 2 == 1:  # 奇数列，水平翻转
                block = torch.flip(block, dims=[3])
            # 放置到对应位置
            underpainting[:, :, i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] = block
    
    return underpainting

def fusion_with_text_image(text_image_t, underpainting_t, threshold=216/255, device=None):
    """
    Module 4: Fusion with Text Image
    将文本图像和对抗水印底纹融合，白色背景区域填充水印底纹
    全部使用torch.Tensor操作
    
    参数：
    - text_image_t: 文本图像 torch.Tensor (1, 3, H, W) BGR格式，范围0-1
    - underpainting_t: 对抗水印底纹 torch.Tensor (1, 3, H', W') BGR格式，范围0-1
    - threshold: 判断白色背景的阈值，默认216/255
    - device: torch device
    
    返回：
    - awti: 对抗水印文本图像 torch.Tensor (1, 3, H, W) BGR格式，范围0-1
    """
    # 确保在正确的设备上
    if device is not None:
        text_image_t = text_image_t.to(device)
        underpainting_t = underpainting_t.to(device)
    
    # 获取文本图像尺寸
    _, _, text_h, text_w = text_image_t.shape
    
    # 如果底纹尺寸小于文本图像，需要平铺或裁剪
    _, _, under_h, under_w = underpainting_t.shape
    
    if under_h < text_h or under_w < text_w:
        # 平铺底纹以匹配文本图像尺寸
        repeat_y = int(torch.ceil(torch.tensor(text_h / under_h)).item())
        repeat_x = int(torch.ceil(torch.tensor(text_w / under_w)).item())
        underpainting_t = underpainting_t.repeat(1, 1, repeat_y, repeat_x)[:, :, :text_h, :text_w]
    else:
        # 裁剪底纹到文本图像尺寸
        underpainting_t = underpainting_t[:, :, :text_h, :text_w]
    
    # 计算RGB总和，判断哪些像素接近白色
    rgb_sum = text_image_t[0, 0, :, :] + text_image_t[0, 1, :, :] + text_image_t[0, 2, :, :]  # (H, W)
    white_mask = rgb_sum >= threshold * 3  # (H, W)
    
    # 创建输出图像（克隆文本图像）
    awti = text_image_t.clone()
    
    # 在白色区域填充水印底纹
    mask_3d = white_mask.unsqueeze(0).expand_as(text_image_t)  # (1, 3, H, W)
    awti[mask_3d] = underpainting_t[mask_3d]
    
    return awti
