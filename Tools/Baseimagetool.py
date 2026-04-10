import os
import numpy as np
import torch
import random
from torchvision import transforms
import torch.nn as nn
from Tools.Showtool import *
import matplotlib.font_manager as fm


def extract_background(img_tensor: torch.Tensor):  #黑色是文字区域，白色是非文字区  提取背景，需要白色区域为1
    assert (len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1)  #(1, 3, 256, 256)
    img_sum = torch.sum(img_tensor, dim=1)  #(1, 256, 256)
    # mask = (img_sum == 3)
    mask = (img_sum > 2.6)
    mask = mask + 0  #将布尔张量转换为数值张量
    return mask.unsqueeze_(0)

def random_image_resize(image: torch.Tensor, low=0.25, high=3):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    #assert image.requires_grad == True
    scale = random.random()
    shape = image.shape
    h, w = shape[-2], shape[-1]
    h, w = int(h * (scale * (high - low) + low)), int(w * (scale * (high - low) + low))
    h, w = h//16*16, w//16*16
    image = transforms.Resize([h, w])(image)
    return image

def repeat_4D(patch: torch.Tensor, h_real, w_real) -> torch.Tensor:
    assert (len(patch.shape) == 4 and patch.shape[0] == 1)
    #assert patch.requires_grad == True
    patch_h,patch_w=patch.shape[2:]
    h_num=h_real//patch_h+1
    w_num = w_real // patch_w+1
    patch = patch.repeat(1, 1, h_num, w_num)
    patch = patch[:, :, :h_real, :w_real]
    return patch

def repeat_2patch(patch1: torch.Tensor, patch2: torch.Tensor, h_real, w_real) -> torch.Tensor:
    # 输入检查
    assert (len(patch1.shape) == 4 and patch1.shape[0] == 1), "patch1 应为 (1, C, H, W) 形状"
    assert (len(patch2.shape) == 4 and patch2.shape[0] == 1), "patch2 应为 (1, C, H, W) 形状"
    
    # 获取通道数和 patch 尺寸
    s = patch1.shape[-1]
    
    # 计算需要的小块数量（向上取整）
    h_num = h_real // s + 1
    w_num = w_real // s + 1
    
    # 初始化目标矩阵，形状为 (1, C, h_num * patch_h, w_num * patch_w)
    patch1 = patch1.repeat(1, 1, h_num, w_num)
    
    # 填充随机比特块
    for r in range(h_num):
        for c in range(w_num):
            # 随机选择 patch1 或 patch2，计算填充位置，填充到 result
            if random.random() < 0.5:
                patch1[:, :, r*s:(r+1)*s, c*s:(c+1)*s] = patch2
    
    # 裁剪到目标尺寸
    patch1 = patch1[:, :, :h_real, :w_real]
    return patch1

def repeat_2patch_random(patch1: torch.Tensor, patch2: torch.Tensor, h_real, w_real) -> torch.Tensor:
    # 输入检查
    assert (len(patch1.shape) == 4 and patch1.shape[0] == 1), "patch1 应为 (1, C, H, W) 形状"
    assert (len(patch2.shape) == 4 and patch2.shape[0] == 1), "patch2 应为 (1, C, H, W) 形状"
    
    # 获取通道数和 patch 尺寸
    C = patch1.shape[1]
    patch_h, patch_w = patch1.shape[2:]
    
    # 计算需要的小块数量（向上取整）
    h_num = (h_real + patch_h - 1) // patch_h
    w_num = (w_real + patch_w - 1) // patch_w
    
    # 初始化目标矩阵，形状为 (1, C, h_num * patch_h, w_num * patch_w)
    result = torch.zeros(1, C, h_num * patch_h, w_num * patch_w, device=patch1.device)
    
    # 填充随机比特块
    for r in range(h_num):
        for c in range(w_num):
            # 随机选择 patch1 或 patch2
            chosen_patch = patch1 if random.random() < 0.5 else patch2
            # 计算填充位置
            start_h = r * patch_h
            start_w = c * patch_w
            end_h = start_h + patch_h
            end_w = start_w + patch_w
            # 填充到 result
            result[:, :, start_h:end_h, start_w:end_w] = chosen_patch
    
    # 裁剪到目标尺寸
    result = result[:, :, :h_real, :w_real]
    return result

def repeat_2patch_flip(patch1: torch.Tensor, patch2: torch.Tensor, h_real, w_real, random_tile=False) -> torch.Tensor:
    # 输入检查
    assert (len(patch1.shape) == 4 and patch1.shape[0] == 1), "patch1 应为 (1, C, H, W) 形状"
    assert (len(patch2.shape) == 4 and patch2.shape[0] == 1), "patch2 应为 (1, C, H, W) 形状"
    
    # 获取通道数和 patch 尺寸
    C = patch1.shape[1]
    patch_h, patch_w = patch1.shape[2:]
    block_h = patch_h * 8  # 水印块高度
    block_w = patch_w * 8  # 水印块宽度

    # 创建 B_patch1 和 B_patch2：将 patch1 和 patch2 平铺到 (1, C, block_h, block_w)
    temp1 = torch.cat([patch1] * 8, dim=2)  # 沿高度重复 8 次
    B_patch1 = torch.cat([temp1] * 8, dim=3)  # 沿宽度重复 8 次
    temp2 = torch.cat([patch2] * 8, dim=2)
    B_patch2 = torch.cat([temp2] * 8, dim=3)

    if random_tile:  # 生成 8×8 的随机选择掩码
        choices = (torch.rand(8, 8, device=patch1.device) < 0.5).float()  # (8, 8)，值为 0 或 1
    else:  # 生成 8×8 的固定选择掩码
        choices = torch.tensor(
            [[0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0]],
            dtype=torch.float32, device=patch1.device
        )
    mask = choices.unsqueeze(0).unsqueeze(0).repeat_interleave(patch_h, dim=2).repeat_interleave(patch_w, dim=3)
    # mask 形状变为 (1, 1, block_h, block_w)，广播后可与 B_patch1 和 B_patch2 运算

    # 创建水印块 B
    B = B_patch1 * mask + B_patch2 * (1 - mask)  # (1, C, block_h, block_w)

    # 生成翻转版本
    B_hflip = torch.flip(B, [3])      # 水平翻转
    B_vflip = torch.flip(B, [2])      # 垂直翻转
    B_hvflip = torch.flip(B, [2, 3])  # 水平和垂直翻转

    # 创建 2×2 模式单元 P
    P_row1 = torch.cat([B, B_hflip], dim=3)      # 第一行：B 和 B_hflip
    P_row2 = torch.cat([B_vflip, B_hvflip], dim=3)  # 第二行：B_vflip 和 B_hvflip
    P = torch.cat([P_row1, P_row2], dim=2)       # (1, C, 2*block_h, 2*block_w)

    # 计算需要重复的模式单元数量
    h_pattern_num = (h_real + 2 * block_h - 1) // (2 * block_h)  # 向上取整
    w_pattern_num = (w_real + 2 * block_w - 1) // (2 * block_w)

    # 平铺 P 到足够大的尺寸
    temp = torch.cat([P] * w_pattern_num, dim=3)  # 沿宽度重复
    large_watermark = torch.cat([temp] * h_pattern_num, dim=2)  # 沿高度重复
    # 形状为 (1, C, 2*block_h*h_pattern_num, 2*block_w*w_pattern_num)

    # 裁剪到目标尺寸
    watermark = large_watermark[:, :, :h_real, :w_real]

    return watermark



def random_noise(image: torch.Tensor,noise_low,noise_high):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    device=image.device
    temp_image=image.clone().detach().cpu().numpy()
    noise=np.random.uniform(low=noise_low,high=noise_high,size=temp_image.shape)
    noise=torch.from_numpy(noise)
    noise=noise.float()
    noise=noise.to(device)

    image=torch.clamp(image+noise,min=0,max=1)
    return image


#CRAFT的操作
def normlize_MeanVariance(image:torch.Tensor,device):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    mean = torch.Tensor([[[[0.485]],[[0.456]],[[ 0.406]]]])
    mean=mean.to(device)
    variance = torch.Tensor([[[[0.229]], [[0.224]], [[0.225]]]])
    variance=variance.to(device)
    image=(image-mean)/variance
    return image

#CRAFT的操作
def resize_aspect_ratio(image:torch.Tensor,square_size,mag_ratio=1.5):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    h,w=image.shape[2:]
    target_size = mag_ratio * max(h, w)
    if target_size>square_size:
        target_size=square_size
    ratio=target_size/max(h,w)
    target_h,target_w=int(h*ratio),int(w*ratio)
    image=transforms.Resize([target_h,target_w])(image)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = torch.zeros([1,3,target_h32, target_w32]).to('cuda:0')
    resized[:,:,0:target_h,0:target_w]=image

    #target_h, target_w = target_h32, target_w32
    #size_heatmap = (int(target_w / 2), int(target_h / 2))
    #return size_heatmap
    return resized,ratio


"""
============================================TEST================================================================
"""


def test_repeat_4D():
    x = torch.Tensor([[[[0.4942, 0.1321],
                          [0.3797, 0.3320]]]])
    x.requires_grad = True
    img_h, img_w = 5, 5
    y = repeat_4D(x, img_h, img_w)

    referance=torch.Tensor([[[[0.4942, 0.1321,0.4942, 0.1321,0.4942],
                              [0.3797, 0.3320,0.3797, 0.3320,0.3797],
                              [0.4942, 0.1321,0.4942, 0.1321,0.4942],
                              [0.3797, 0.3320,0.3797, 0.3320,0.3797],
                              [0.4942, 0.1321,0.4942, 0.1321,0.4942],]]])
    assert (y==referance).all()
    img_grad_show(y)

def test_random_resize():
    from Tools.Showtool import img_grad_show
    img = torch.randn(1, 3, 120, 100)
    img.requires_grad = True
    img = random_image_resize(img, low=0.1, high=3)
    img_grad_show(img)

def test_resize_aspect_ratio():
    from Tools.Showtool import img_grad_show
    img=torch.randn(1,3,5,5)
    img.requires_grad=True
    img=img.cuda()
    resize_image,ratio=resize_aspect_ratio(image=img,device=torch.device('cuda:0'),
                                           square_size=10,mag_ratio=1.5)
    print(resize_image.shape)
    print(resize_image[0,0,:10,0])
    img_grad_show(resize_image)


"""
        ==========================================
        =========== general function =============
        ==========================================
"""

# 定义八种状态的变换及其逆变换
TRANSFORMS = {
    1: lambda x: x,  # 原始状态
    2: lambda x: cv2.flip(x, 1),  # 水平翻转
    3: lambda x: cv2.flip(x, 0),  # 垂直翻转
    4: lambda x: cv2.flip(cv2.flip(x, 1), 0),  # 水平和垂直翻转
    5: lambda x: cv2.flip(cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE), 1),  # 顺时针90度 + 水平翻转
    6: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),  # 顺时针旋转90度
    7: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),  # 逆时针旋转90度
    8: lambda x: cv2.flip(cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)  # 逆时针90度 + 水平翻转
}

INVERSE_TRANSFORMS = {
    1: lambda x: x,
    2: lambda x: cv2.flip(x, 1),
    3: lambda x: cv2.flip(x, 0),
    4: lambda x: cv2.flip(cv2.flip(x, 1), 0),
    5: lambda x: cv2.rotate(cv2.flip(x, 1), cv2.ROTATE_90_COUNTERCLOCKWISE),
    6: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
    7: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
    8: lambda x: cv2.rotate(cv2.flip(x, 1), cv2.ROTATE_90_CLOCKWISE)
}

def apply_inverse_transform(tensor, state, device=None):
    """
    对torch tensor应用逆变换
    tensor: [C, H, W] RGB [0,1] 或 [H, W]
    state: 1-8
    device: torch device (兼容性参数，不使用)
    """
    if state == 1:
        return tensor
    elif state == 2:
        return torch.flip(tensor, dims=[2])  # 水平翻转 [C,H,W] -> flip W
    elif state == 3:
        return torch.flip(tensor, dims=[1])  # 垂直翻转 [C,H,W] -> flip H
    elif state == 4:
        return torch.flip(torch.flip(tensor, dims=[2]), dims=[1])  # 水平+垂直
    elif state == 5:
        # 顺时针90度 + 水平翻转 -> 逆: 水平翻转 + 逆时针90度
        rotated = torch.rot90(tensor, k=1, dims=[1, 2])  # 逆时针90 [C,H,W] -> rotate on H,W
        return torch.flip(rotated, dims=[2])
    elif state == 6:
        return torch.rot90(tensor, k=1, dims=[1, 2])  # 逆时针90度
    elif state == 7:
        return torch.rot90(tensor, k=-1, dims=[1, 2])  # 顺时针90度
    elif state == 8:
        # 逆时针90度 + 水平翻转 -> 逆: 水平翻转 + 顺时针90度
        rotated = torch.rot90(tensor, k=-1, dims=[1, 2])  # 顺时针90
        return torch.flip(rotated, dims=[2])
    else:
        return tensor

def gen_iter_dict(root_eval,special_eval_item):
    mypath=os.path.join(root_eval,special_eval_item)
    dir_list=os.listdir(mypath)
    is_need=[]
    for item in dir_list:
        if item.isdigit():
            is_need.append(item)
    return is_need

def delete_scaled_files(folder_path):
    """
    删除指定文件夹下所有文件名以'_scaled.png'结尾的文件。
    
    参数:
        folder_path (str): 用户提供的文件夹路径
    
    返回:
        None
    """
    # 检查文件夹路径是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在！")
        return
    
    # 检查文件夹路径是否是一个目录
    if not os.path.isdir(folder_path):
        print(f"错误：'{folder_path}' 不是一个文件夹！")
        return
    
    # 统计符合条件的文件
    files_to_delete = [f for f in os.listdir(folder_path) if f.endswith(('_scaled.png', '_cropped.png', '_ti_line.png',
        '_symmetry.png', '_symmetry_normalized.png', '_corner_points.png', '_rectangle.png', '_awu_hat.png',
        '_histogram.png', '_ti_hat.png', '_restored_wmblock.png', '_transformed.png', '_detect.png',))]
    
    if not files_to_delete:
        print(f"文件夹 '{folder_path}' 无需清理。")
        return
    
    # 执行删除操作并处理可能的错误
    for file in files_to_delete:
        file_path = os.path.join(folder_path, file)
        try:
            os.remove(file_path)
            print(f"已删除：{file}")
        except FileNotFoundError:
            print(f"错误：文件 '{file}' 不存在，可能已被删除。")
        except PermissionError:
            print(f"错误：无权限删除文件 '{file}'。")
        except Exception as e:
            print(f"删除文件 '{file}' 时发生未知错误：{e}")


# 设置中文字体 - 自动检测可用的中文字体
def get_chinese_font():
    """获取系统中可用的中文字体"""
    chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
                     'Noto Sans CJK SC', 'Noto Sans CJK TC', 'AR PL UMing CN',
                     'AR PL UKai CN', 'Microsoft YaHei', 'STHeiti', 'STSong']
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            print(f"使用中文字体: {font}")
            return font
    
    # 如果没有找到中文字体，使用支持Unicode的字体
    print("未找到中文字体，使用DejaVu Sans")
    return 'DejaVu Sans'