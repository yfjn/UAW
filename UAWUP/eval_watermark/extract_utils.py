import cv2
import numpy as np
from collections import Counter
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/UAW/")
from Tools.Baseimagetool import *
from Tools.ImageIO import *


def extract_text_mask(image_t, morph_open, save_path, f, device=None, otsu=False):
    """
    提取图像中的黑色文字掩码，使用固定阈值0.604 (154/255)
    
    参数：
    - image_t: 输入的彩色图像 torch.Tensor [C,H,W]，范围 [0,1]，RGB格式
    - morph_open: 是否使用开运算进行处理
    - save_path: 保存路径
    - f: 日志函数句柄
    - device: torch device
    
    返回：
    - text_mask: 文字掩码 torch.Tensor (H, W)，1 表示黑色文字，0 表示非文字
    """
    if otsu == False:
        # 转换为灰度图像 (weighted sum: 0.299R + 0.587G + 0.114B)
        # image_t是[C,H,W] RGB格式，范围[0,1]
        gray = 0.299 * image_t[0] + 0.587 * image_t[1] + 0.114 * image_t[2]  # [H,W]
        
        # 固定阈值 (154/255 = 0.604)
        fixed_thresh = 0.604
        
        # 二值化：像素值 <= 阈值 → 1 (文字)，否则 → 0 (背景)
        binary = (gray <= fixed_thresh).float()
        
        # 保存直方图（需要转到CPU并转换为0-255范围）
        gray_255 = (gray * 255.0).cpu().numpy().astype(np.uint8)
        hist = cv2.calcHist([gray_255], [0], None, [256], [0, 256])
        fig = plt.figure(figsize=(10, 6))
        plt.plot(hist)
        plt.title('Grayscale Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Pixel Count')
        plt.axvline(x=154, color='r', linestyle='--', label=f'Fixed threshold: 154 (0.604)')
        plt.legend()
        hist_path = save_path + '_gray_histogram.png'
        plt.savefig(hist_path)
        logINFO(f"{hist_path} is saved with fixed threshold 154 (0.604), pixels below this value are considered as text", f)
        plt.close(fig)
        
        if morph_open:
            # 开运算（腐蚀后膨胀）- 使用torch操作
            import torch.nn.functional as F
            # 定义3x1内核
            kernel = torch.ones((1, 1, 3, 1), device=device if device else torch.device('cpu'))  # (out_ch, in_ch, h, w)
            binary_4d = binary.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            # 腐蚀: min pooling
            eroded = -F.max_pool2d(-binary_4d, kernel_size=(3, 1), stride=1, padding=(1, 0))
            # 膨胀: max pooling
            opened = F.max_pool2d(eroded, kernel_size=(3, 1), stride=1, padding=(1, 0))
            
            binary = opened.squeeze(0).squeeze(0)
        
        # 转换为0和1的掩码
        text_mask = (binary > 0.5).to(torch.uint8)
        
        return text_mask

    else:
        """
        提取图像中的黑色文字掩码，使用大津算法自动确定最优阈值。物理失真时用
        全部使用torch.Tensor操作，支持GPU加速

        参数：
        - image_t: 输入的彩色图像 torch.Tensor [C,H,W]，范围 [0,1]，RGB格式
        - morph_open: 是否使用开运算进行处理
        - save_path: 保存路径
        - f: 日志函数句柄
        - device: torch device

        返回：
        - text_mask: 文字掩码 torch.Tensor (H, W)，1 表示黑色文字，0 表示非文字
        """
        # 转换为灰度图像 (weighted sum: 0.299R + 0.587G + 0.114B)
        # image_t是[C,H,W] RGB格式，范围[0,1]
        gray = 0.299 * image_t[0] + 0.587 * image_t[1] + 0.114 * image_t[2]  # [H,W]
        
        # 将灰度值转换到0-255范围用于Otsu计算
        gray_255 = (gray * 255.0).clamp(0, 255)
        
        # 使用PyTorch实现Otsu算法
        def otsu_threshold_torch(gray_img):
            """
            使用PyTorch实现Otsu阈值算法
            gray_img: [H,W] tensor，范围0-255
            返回: 最优阈值 (0-255范围)
            """
            # 计算直方图
            hist = torch.histc(gray_img, bins=256, min=0, max=255)
            total_pixels = gray_img.numel()
            
            # 归一化直方图得到概率分布
            prob = hist / total_pixels
            
            # 计算累积和
            bin_centers = torch.arange(256, dtype=torch.float32, device=gray_img.device)
            
            # 类间方差计算
            weight_bg = torch.cumsum(prob, dim=0)  # 背景权重
            weight_fg = 1.0 - weight_bg  # 前景权重
            
            # 累积均值
            mean_bg = torch.cumsum(prob * bin_centers, dim=0) / (weight_bg + 1e-10)
            
            # 总均值
            total_mean = torch.sum(prob * bin_centers)
            
            # 前景均值
            mean_fg = (total_mean - weight_bg * mean_bg) / (weight_fg + 1e-10)
            
            # 类间方差
            variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            
            # 找到最大类间方差对应的阈值
            optimal_thresh = torch.argmax(variance_between).item()
            
            return optimal_thresh
        
        # 计算Otsu阈值
        otsu_thresh = otsu_threshold_torch(gray_255)
        
        # 固定阈值参考 (254 - eps = 154)
        eps = 100
        fixed_thresh = 254 - eps
        
        # 计算并保存直方图（转到CPU进行绘图）
        gray_255_cpu = gray_255.cpu().numpy().astype(np.uint8)
        hist = cv2.calcHist([gray_255_cpu], [0], None, [256], [0, 256])
        fig = plt.figure(figsize=(10, 6))
        plt.plot(hist)
        plt.title('Grayscale Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Pixel Count')
        
        # 在直方图上标记阈值
        plt.axvline(x=otsu_thresh, color='r', linestyle='--', label=f'Otsu threshold: {otsu_thresh}')
        plt.axvline(x=fixed_thresh, color='g', linestyle=':', label=f'Fixed threshold: {fixed_thresh}')
        plt.legend()
        
        # 保存直方图
        plt.savefig(save_path + '_gray_histogram.png')
        plt.close(fig)
        
        # 二值化：像素值 <= 阈值 → 1 (文字)，否则 → 0 (背景)
        # 使用Otsu阈值，转换回0-1范围
        otsu_thresh_normalized = otsu_thresh / 255.0
        binary = (gray <= otsu_thresh_normalized).float()
        
        if morph_open:
            # 开运算（腐蚀后膨胀）- 使用torch操作
            binary_4d = binary.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            # 腐蚀: min pooling
            eroded = -F.max_pool2d(-binary_4d, kernel_size=(3, 1), stride=1, padding=(1, 0))
            # 膨胀: max pooling
            opened = F.max_pool2d(eroded, kernel_size=(3, 1), stride=1, padding=(1, 0))
            
            binary = opened.squeeze(0).squeeze(0)
        
        # 转换为0和1的掩码
        text_mask = (binary > 0.5).to(torch.uint8)
        
        return text_mask


def create_text_image(image_t, text_mask_t, device=None):
    """
    创建文字图像：文字部分保留原始图像的像素值，其余部分为白色（1.0）
    
    参数：
    - image_t: 输入的彩色图像 torch.Tensor [C,H,W]，范围 [0,1]，RGB格式
    - text_mask_t: 文字掩码 torch.Tensor [H,W]，1 表示文字，0 表示非文字
    - device: torch device
    
    返回：
    - text_image: 文字图像 torch.Tensor [C,H,W]，范围 [0,1]，RGB格式
    """
    # 创建全白背景
    text_image = torch.ones_like(image_t)
    
    # 将文字部分设置为原始图像的像素值
    mask_3d = text_mask_t.unsqueeze(0).expand_as(image_t)  # [H,W] -> [C,H,W]
    text_image[mask_3d == 1] = image_t[mask_3d == 1]
    
    return text_image

def calculate_symmetry_mask(image_t, text_mask_t, img_type='awti', direction='column', save_path='eval_watermark/img', device=None):
    """
    计算图像的行或列对称性信号，利用文字掩码排除文字区域影响
    全部使用torch.Tensor操作
    
    参数：
    - image_t: 输入的RGB图像tensor [C,H,W]，范围 [0,1]
    - text_mask_t: 文字掩码tensor [H,W]，1 表示文字，0 表示非文字
    - img_type: 图像类型
    - direction: 'column' 或 'row'
    - save_path: 保存路径
    - device: torch device (None表示CPU)
    
    返回：
    - symmetry_signal: 对称性信号numpy数组 (W 或 H)
    """
    # 确保在正确的设备上
    if device is not None:
        image_t = image_t.to(device)
        text_mask_t = text_mask_t.to(device)
    
    c, m, n = image_t.shape
    signal_len = n if direction == 'column' else m
    symmetry_signal = torch.zeros(signal_len, dtype=torch.float32, device=device if device else torch.device('cpu'))
    
    if direction == 'column':
        for j in range(n):
            dj = min(j, n - j)
            if dj == 0:
                symmetry_signal[j] = 0
                continue
            
            J1 = image_t[:, :, j - dj:j]  # [C, H, dj]
            J2 = image_t[:, :, j:j + dj]  # [C, H, dj]
            mask_J1 = text_mask_t[:, j - dj:j]  # [H, dj]
            mask_J2 = text_mask_t[:, j:j + dj]  # [H, dj]
            
            J2_flipped = torch.flip(J2, dims=[2])  # 水平翻转
            C1 = (J1 - torch.mean(J1)) / (torch.std(J1) + 1e-10)
            C2 = (J2_flipped - torch.mean(J2_flipped)) / (torch.std(J2_flipped) + 1e-10)
            
            # 扩展mask到3通道 [H,dj] -> [C,H,dj]
            non_text_mask_J1 = (mask_J1 == 0).float().unsqueeze(0).expand(c, -1, -1)
            non_text_mask_J2 = (mask_J2 == 0).float().unsqueeze(0).expand(c, -1, -1)
            non_text_mask_J2_flipped = torch.flip(non_text_mask_J2, dims=[2])
            
            C1_non_text = C1 * non_text_mask_J1
            C2_non_text = C2 * non_text_mask_J2_flipped
            
            symmetry_signal[j] = torch.sum(C1_non_text * C2_non_text) / (c * m * dj + 1e-10)
    else:  # row
        for i in range(m):
            di = min(i, m - i)
            if di == 0:
                symmetry_signal[i] = 0
                continue
            
            I1 = image_t[:, i - di:i, :]  # [C, di, W]
            I2 = image_t[:, i:i + di, :]  # [C, di, W]
            mask_I1 = text_mask_t[i - di:i, :]  # [di, W]
            mask_I2 = text_mask_t[i:i + di, :]  # [di, W]
            
            I2_flipped = torch.flip(I2, dims=[1])  # 垂直翻转
            C1 = (I1 - torch.mean(I1)) / (torch.std(I1) + 1e-10)
            C2 = (I2_flipped - torch.mean(I2_flipped)) / (torch.std(I2_flipped) + 1e-10)
            
            # 扩展mask到3通道 [di,W] -> [C,di,W]
            non_text_mask_I1 = (mask_I1 == 0).float().unsqueeze(0).expand(c, -1, -1)
            non_text_mask_I2 = (mask_I2 == 0).float().unsqueeze(0).expand(c, -1, -1)
            non_text_mask_I2_flipped = torch.flip(non_text_mask_I2, dims=[1])
            
            C1_non_text = C1 * non_text_mask_I1
            C2_non_text = C2 * non_text_mask_I2_flipped
            
            symmetry_signal[i] = torch.sum(C1_non_text * C2_non_text) / (c * di * n + 1e-10)
    
    # 转换回numpy返回
    return symmetry_signal.cpu().numpy()

def filter_peaks(symmetry_signal, num_parts, num_candidates, f):
    """
    在每段中遍历每个点，计算其与相邻点的差值得分（边界点仅单侧），
    每段选出得分最高的点，最终保留全局 top-k。

    参数：
    - symmetry_signal: 对称性信号
    - num_parts: 分段数量
    - num_candidates: 最终保留的点数
    - f: 日志句柄（可选）

    返回：
    - peak_positions: 得分最高的点的位置列表
    """
    part_size = len(symmetry_signal) // num_parts
    peaks = []
    scores = []

    for i in range(num_parts):
        start = i * part_size
        end = (i + 1) * part_size if i < num_parts - 1 else len(symmetry_signal)

        local_best_score = -np.inf
        local_best_pos = -1

        for j in range(start, end):
            if j == 0:
                score = symmetry_signal[j] - symmetry_signal[j + 1]
            elif j == len(symmetry_signal) - 1:
                score = symmetry_signal[j] - symmetry_signal[j - 1]
            else:
                left_diff = symmetry_signal[j] - symmetry_signal[j - 1]
                right_diff = symmetry_signal[j] - symmetry_signal[j + 1]
                score = (left_diff + right_diff) / 2
            # logINFO((i, j, score), f)

            if score > local_best_score:
                local_best_score = score
                local_best_pos = j

        if local_best_pos != -1:
            peaks.append(local_best_pos)
            scores.append(local_best_score)
            # logINFO((i, start, end, local_best_pos, local_best_score), f)

    # 全局排序并保留 top-k
    sorted_indices = np.argsort(scores)[::-1]
    peak_positions = [peaks[i] for i in sorted_indices[:num_candidates]]
    return sorted(peak_positions)

def save_symmetry_signal(column_symmetry, row_symmetry, save_path, language='zh'):
    """
    保存列和行对称性信号图像，并在同一图中左右分布显示。

    参数：
    - column_symmetry: 列对称性信号（一维数组或列表）
    - row_symmetry: 行对称性信号（一维数组或列表）
    - save_path: 保存路径前缀（不包含文件名后缀）
    """
    # plt.figure(figsize=(14, 6))
    plt.figure(figsize=(12, 6))

    # 可选：全局字体调整
    if language == 'zh':
        plt.rcParams['font.family'] = get_chinese_font()
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams.update({'font.size': 16})

    # 左图：列对称性
    # plt.subplot(1, 2, 1)
    plt.plot(column_symmetry)
    if language == 'en':
        plt.title('Symmetry Signal (Column)', fontsize=20)
        plt.xlabel('Index', fontsize=18)
        plt.ylabel('Zero-centered Symmetry Value', fontsize=18)
    elif language == 'zh':
        plt.title('对称信号（列）', fontsize=20)
        plt.xlabel('索引', fontsize=18)
        plt.ylabel('零中心化对称值', fontsize=18)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=16)

    # 右图：行对称性
    # plt.subplot(1, 2, 2)
    # plt.plot(row_symmetry)
    # plt.title('Symmetry Signal (Row)', fontsize=20)
    # plt.xlabel('Index', fontsize=18)
    # plt.ylabel('Zero-centered Symmetry Value', fontsize=18)
    # plt.grid(True)
    # plt.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.savefig(save_path + '_awti_symmetry.png', dpi=150)
    # plt.savefig(save_path + '_awti_symmetry.pdf', format='pdf')
    plt.close()

def compute_similarity(block, patch, device=None):
    """计算块与 Patch 的相似度（归一化互相关）"""
    if device is not None and torch.is_tensor(block) and torch.is_tensor(patch):
        # 使用内置 corrcoef，底层已高度优化
        stacked = torch.stack([block.flatten(), patch.flatten()])
        return torch.corrcoef(stacked)[0, 1].item()
    else:
        return np.corrcoef(block.flatten(), patch.flatten())[0, 1]

def determine_bit_block_state(bit_block, p_0, p_1, device=None):
    """确定单个比特块的状态，基于八种状态下与 p_0 和 p_1 的最大 NCC
    支持torch.Tensor和numpy.ndarray
    """
    best_state = 1
    best_similarity = -1
    for state in range(1, 9):  # 考虑八种状态
        restored = apply_inverse_transform(bit_block, state, device)
        sim0 = compute_similarity(restored, p_0, device=device)
        sim1 = compute_similarity(restored, p_1, device=device)
        max_sim = max(sim0, sim1)
        if max_sim > best_similarity:
            best_similarity = max_sim
            best_state = state
    return best_state

def infer_all_watermark_blocks(first_block_y, first_block_x, states, s, wm_h=8, wm_w=8, device=None):
    """
    从第一个水印块的最左上角网格位置开始，以步长 wm_h, wm_w 推断所有完整的水印块。
    支持torch.Tensor和numpy.ndarray

    参数:
        first_block_y, first_block_x: 第一个水印块的左上角比特块坐标
        states: 比特块状态矩阵 (rows, cols) - torch.Tensor或numpy.ndarray
        s: 比特块大小
        device: torch device (仅torch模式使用)

    返回:
        watermark_blocks: 所有完整水印块的列表 [(block_y, block_x, state), ...]
    """
    rows, cols = states.shape
    watermark_blocks = []
    
    # 计算网格的最左上角起始位置
    start_y = first_block_y % wm_h
    start_x = first_block_x % wm_w
    
    # 从最左上角以步长 wm_h, wm_w 遍历
    for i in range(start_y, rows - wm_h + 1, wm_h):
        for j in range(start_x, cols - wm_w + 1, wm_w):
            region = states[i:i+wm_h, j:j+wm_w]
            # 使用torch进行统计
            state_counts = torch.bincount(region.flatten().long(), minlength=9)
            majority_state = torch.argmax(state_counts[1:]).item() + 1
            watermark_blocks.append((i, j, majority_state))
    
    return watermark_blocks

def extract_bit_from_block(bit_block, p_0, p_1, device=None):
    """从恢复后的比特块中提取水印位
    支持torch.Tensor和numpy.ndarray
    """
    sim0 = compute_similarity(bit_block, p_0, device=device)
    sim1 = compute_similarity(bit_block, p_1, device=device)
    # sim0 = cv2.matchTemplate(bit_block, p_0, cv2.TM_CCOEFF_NORMED)[0][0]
    # sim1 = cv2.matchTemplate(bit_block, p_1, cv2.TM_CCOEFF_NORMED)[0][0]
    return 0 if sim0 > sim1 else 1


"""
        ==========================================
        ============= Module Functions ===========
        ==========================================
"""

def watermark_block_synchronization(awti_img, s, num_repeats=4, wm_h=8, wm_w=8, save_path=None, f=None, device=None, phy=False):
    """
    Module 1: Watermark Block Synchronization - 水印块同步，检测对称轴并定位水印块
    
    参数：
    - awti_img: 对抗水印文本图像 torch.Tensor [C,H,W] RGB格式，范围[0,1]
    - s: 比特块尺寸
    - num_repeats: 平铺次数，默认5
    - wm_h: 水印高度，默认8
    - wm_w: 水印宽度，默认8
    - save_path: 保存路径（可选，用于日志和调试图像保存）
    - f: 日志文件句柄（可选）
    - device: torch device，默认None表示CPU
    
    返回：
    - col_peaks: 列对称轴峰值位置列表
    - row_peaks: 行对称轴峰值位置列表
    - wh: 比特块宽高 (w, h)
    - xy0: 最左上角比特块位置 (x0, y0)
    - XY0: 最左上角水印块位置 (X0, Y0)
    """
    ##################################
    #########   对称性补偿    #########
    ##################################
    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取对抗水印文档图像
    if awti_img is not None:
        # 从tensor读取
        if device is not None:
            awti_img = awti_img.to(device)
        # awti_img已经是[C,H,W] RGB [0,1]格式
        aw_image_t = awti_img
    else:
        # 从文件读取 (img_read返回[1,C,H,W])
        aw_image_t = img_read(save_path, device=device).squeeze(0)
        
    C, H, W = aw_image_t.shape
    
    # 提取文字掩码和创建文字图像（全部使用tensor）
    text_mask_t = extract_text_mask(aw_image_t, False, save_path, f, device=device, otsu=(phy == True))  # (H,W) 1表示文字，0表示非文字
    text_image_t = create_text_image(aw_image_t, text_mask_t, device=device)
    
    # 保存文字图像（需要转换到CPU和numpy，并转换为BGR格式）
    text_image_np = (text_image_t * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [C,H,W]->[H,W,C] 0-255
    text_image_np = cv2.cvtColor(text_image_np, cv2.COLOR_RGB2BGR)  # RGB->BGR for cv2
    cv2.imwrite(save_path + '_ti_hat.png', text_image_np)

    if phy:
        return (0, 0), (0, 0)
    
    # 计算对称性信号（使用tensor）
    s_column_opt = calculate_symmetry_mask(aw_image_t, text_mask_t, 'awti', 'column', save_path, device=device)
    s_row_opt = calculate_symmetry_mask(aw_image_t, text_mask_t, 'awti', 'row', save_path, device=device)
    save_symmetry_signal(s_column_opt, s_row_opt, save_path)
    
    # 判断 num_repeats 是单值还是元组/列表
    if isinstance(num_repeats, (tuple, list)):
        col_repeat = num_repeats[0]
        row_repeat = num_repeats[1]
    elif num_repeats == 0:
        num_repeats = W//s//8, H//s//8
        col_repeat, row_repeat = num_repeats
    else:
        col_repeat = row_repeat = num_repeats

    # 滤波峰值
    col_peaks = filter_peaks(s_column_opt, col_repeat + 1, col_repeat + 1, f)
    row_peaks = filter_peaks(s_row_opt, row_repeat + 1, row_repeat + 1, f)
    logINFO(f"检测到的列对称轴: {col_peaks}", f)
    logINFO(f"检测到的行对称轴: {row_peaks}", f)
    
    ##################################
    #########   分割水印块    #########
    ##################################
    # 推断块大小 (find_block_size_auto内容)
    if len(col_peaks) < 2 or len(row_peaks) < 2:
        raise ValueError("角点数量不足，无法推断块大小")
    
    x_diffs = [col_peaks[i+1] - col_peaks[i] for i in range(len(col_peaks)-1) if col_peaks[i+1] - col_peaks[i] > 2]
    y_diffs = [row_peaks[i+1] - row_peaks[i] for i in range(len(row_peaks)-1) if row_peaks[i+1] - row_peaks[i] > 2]
    
    if not x_diffs or not y_diffs:
        raise ValueError("无法推断块大小：所有角点在同一行或同一列")
    
    block_width = Counter(x_diffs).most_common(1)[0][0]
    block_height = Counter(y_diffs).most_common(1)[0][0]
    logINFO(f'x方向非零差值频率(前10个):{Counter(x_diffs).most_common(10)}', f)
    logINFO(f'y方向非零差值频率(前10个):{Counter(y_diffs).most_common(10)}', f)
    logINFO(f"推断的水印块大小w*h: {block_width}x{block_height}", f)
    
    # 确定最左上角水印块的起始位置 (find_top_left_position_auto内容)
    x_mods = [x % block_width for x in col_peaks]
    y_mods = [y % block_height for y in row_peaks]
    
    x_freq = Counter(x_mods)
    y_freq = Counter(y_mods)
    logINFO(f'x方向取模后值(前10个):{x_freq.most_common(10)}', f)
    logINFO(f'y方向取模后值(前10个):{y_freq.most_common(10)}', f)
    
    max_x_freq = max(x_freq.values())
    x_candidates = [k for k, v in x_freq.items() if v == max_x_freq]
    X0 = min(x_candidates)
    
    max_y_freq = max(y_freq.values())
    y_candidates = [k for k, v in y_freq.items() if v == max_y_freq]
    Y0 = min(y_candidates)
    logINFO(f"divide最左上角水印块的起始位置: ({X0}, {Y0})", f)

    # 计算缩放比例
    if block_height != s*wm_h or block_width != s*wm_w:
        scale_factor_h = s*wm_h / block_height
        scale_factor_w = s*wm_w / block_width
        
        # 使用torch的interpolate进行缩放，保持在GPU上
        import torch.nn.functional as F
        new_height = round(H * scale_factor_h)
        new_width = round(W * scale_factor_w)
        
        # aw_image_t是[C,H,W]，需要转为[1,C,H,W]
        img_t = aw_image_t.unsqueeze(0)
        img_t = F.interpolate(img_t, size=(new_height, new_width), mode='bilinear', align_corners=False)
        scaled_image_t = img_t.squeeze(0)  # [C,H,W]
        
        # 转回numpy保存 (RGB->BGR)
        scaled_image_np = (scaled_image_t * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        scaled_image_np = cv2.cvtColor(scaled_image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path + "_awti_scaled.png", scaled_image_np)
        
        block_width, block_height = s*wm_h, s*wm_w
        X0 = round(X0 * scale_factor_w)
        Y0 = round(Y0 * scale_factor_h)
        logINFO(f"发现水印块经历了缩放，图像已缩放至 {new_width}x{new_height}", f)
        logINFO(f"水印块尺寸变为 {block_width}x{block_height}，后续将使用缩放后的文本图像", f)
    
    # 确定最左上角比特块的起始位置
    x0, y0 = X0, Y0
    while x0 >= s:
        x0 -= s
    while y0 >= s:
        y0 -= s
    xy0 = (x0, y0)  # 最左上角比特块位置
    XY0 = (X0, Y0)  # 最左上角水印块位置
    
    return xy0, XY0

def watermark_block_state_determination(awti_img, s, xy0, XY0, wm_h=8, wm_w=8, save_path=None, f=None, device=None):
    """
    Module 2: Watermark Block State Determination - 确定水印块状态并恢复水印底纹
    
    参数：
    - awti_img: 对抗水印文本图像 torch.Tensor (H, W, 3) BGR格式
    - s: 比特块尺寸
    - xy0: 最左上角比特块位置 (x0, y0)
    - XY0: 最左上角水印块位置 (X0, Y0)
    - wm_h: 水印高度
    - wm_w: 水印宽度
    - awti_img: 水印文本图像 torch.Tensor 或 None (从文件读取)
    - device: torch device (None表示CPU)
    
    返回：
    - restored_image_t: 恢复的水印底纹图像 torch.Tensor (H, W, C), RGB, 0-255
    """
    ##################################
    ########   重构水印图像    ########
    ##################################
    x0, y0 = xy0
    X0, Y0 = XY0
    X0, Y0 = X0//s, Y0//s
    
    # 读取图像
    if os.path.exists(save_path + "_awti_scaled.png"):
        # 缩放后的图像存在，加载它
        image_t = img_read(save_path + "_awti_scaled.png", device=device).squeeze(0)  # [C,H,W] RGB [0,1]
    else:
        if awti_img is not None:
            # 从tensor转换，保持在GPU上
            if device is not None:
                awti_img = awti_img.to(device)
            # awti_img已经是[C,H,W] RGB [0,1]格式
            image_t = awti_img
        else:
            # 从文件读取
            image_t = img_read(save_path + "_awti.png", device=device).squeeze(0)  # [C,H,W] RGB [0,1]
    
    # 根据 (x0, y0) 裁剪图像
    c, height, width = image_t.shape
    cropped_tensor = image_t[:, y0:, x0:]  # [C, H', W'] 保持在GPU上
    cropped_height, cropped_width = cropped_tensor.shape[1], cropped_tensor.shape[2]
    
    # 保存裁剪后的图像（仅用于调试）
    cropped_np = (cropped_tensor * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [C,H,W]->[H,W,C]
    cropped_np = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR)  # RGB->BGR
    cv2.imwrite(save_path + '_awti_cropped.png', cropped_np)
    
    # 计算比特块的行数和列数
    rows = cropped_height // s
    cols = cropped_width // s
    
    # 使用tensor操作，全部在GPU上
    new_tensor = cropped_tensor.clone()
    
    # 创建掩码矩阵
    mask = torch.zeros(rows, dtype=torch.uint8, device=device if device else torch.device('cpu'))
    
    # 根据 Y0 初始化掩码
    for i in range(rows):
        if Y0 % 2:
            mask[i] = 1 if i % 2 == 1 else 0
        else:
            mask[i] = 1 if i % 2 == 0 else 0
    
    # 遍历所有行
    for i in range(rows):
        flip_i = i + (wm_h*2-1 - ((i - Y0) % wm_h) * 2)
        if flip_i < 0 or flip_i >= rows:
            continue
        
        if mask[i] == 1 and mask[flip_i] == 0:
            start_x = flip_i * s
            flip_start_x = i * s
            # 使用torch.flip进行翻转 [C,H,W]
            flipped_row = torch.flip(cropped_tensor[:, flip_start_x:flip_start_x+s, :], dims=[1])  # 垂直翻转
            new_tensor[:, start_x:start_x+s, :] = flipped_row
            mask[flip_i] = 1
        
        elif mask[i] == 0 and mask[flip_i] == 1:
            start_x = i * s
            flip_start_x = flip_i * s
            flipped_row = torch.flip(cropped_tensor[:, flip_start_x:flip_start_x+s, :], dims=[1])
            new_tensor[:, start_x:start_x+s, :] = flipped_row
            mask[i] = 1
    
    # 保存重建后的图像 (RGB to BGR)
    new_image_bgr = (new_tensor * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    new_image_bgr = cv2.cvtColor(new_image_bgr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path + '_awu_hat.png', new_image_bgr)
    
    # 返回恢复的水印底纹tensor [C,H,W] RGB [0,1]
    return new_tensor

def watermark_message_extraction(input_img, p_0, p_1, s, xy0, XY0, true_watermark, wm_h=8, wm_w=8, f=None, device=None):
    """
    Module 3: Watermark Message Extraction - 从图像中提取水印信息
    
    参数：
    - input_img: 输入图像 torch.Tensor [C,H,W] RGB [0,1]，可以是awti_cropped或awu_hat
    - p_0: 比特模板0 torch.Tensor [C,H,W] RGB [0,1]
    - p_1: 比特模板1 torch.Tensor [C,H,W] RGB [0,1]
    - s: 比特块尺寸
    - xy0: 最左上角比特块位置 (x0, y0)
    - XY0: 最左上角水印块位置 (X0, Y0)
    - true_watermark: 真实水印矩阵 (wm_h, wm_w) numpy或tensor
    - wm_h: 水印高度，默认8
    - wm_w: 水印宽度，默认8
    - save_path: 保存路径（可选）
    - f: 日志文件句柄（可选）
    - device: torch device，默认None表示CPU
    
    返回：
    - accuracy: 比特正确率
    """
    # 确保在正确的设备上
    if device is None:
        device = torch.device('cpu')
    
    # 直接使用[C,H,W] RGB [0,1]格式
    image_t = input_img.to(device)
    p_0_t = p_0.to(device)
    p_1_t = p_1.to(device)
    
    # 计算比特块的行数和列数
    c, height, width = image_t.shape
    rows = height // s
    cols = width // s
    
    # 批量提取所有比特块 - 使用unfold进行高效提取
    # unfold: [C,H,W] -> [C, rows, cols, s, s]
    blocks = image_t.unfold(1, s, s).unfold(2, s, s)  # [C, rows, cols, s, s]
    blocks = blocks.permute(1, 2, 0, 3, 4)  # [rows, cols, C, s, s]
    
    # 扁平化模板以加速相似度计算
    p_0_flat = p_0_t.flatten()  # [C*s*s]
    p_1_flat = p_1_t.flatten()  # [C*s*s]
    p_0_std = torch.std(p_0_flat)
    p_1_std = torch.std(p_1_flat)
    p_0_mean = torch.mean(p_0_flat)
    p_1_mean = torch.mean(p_1_flat)
    p_0_normalized = (p_0_flat - p_0_mean) / (p_0_std + 1e-10)
    p_1_normalized = (p_1_flat - p_1_mean) / (p_1_std + 1e-10)
    
    # 批量判断所有比特块的状态
    states = torch.zeros((rows, cols), dtype=torch.int32, device=device)
    blocks_flat = blocks.reshape(rows, cols, -1)  # [rows, cols, C*s*s]
    
    for i in range(rows):
        for j in range(cols):
            block_flat = blocks_flat[i, j]  # [C*s*s]
            best_state = 1
            best_similarity = -1
            
            for state in range(1, 9):
                # 应用逆变换
                block_3d = blocks[i, j]  # [C, s, s]
                restored = apply_inverse_transform(block_3d, state, device=device)
                restored_flat = restored.flatten()
                
                # 快速计算NCC
                restored_std = torch.std(restored_flat)
                restored_mean = torch.mean(restored_flat)
                restored_normalized = (restored_flat - restored_mean) / (restored_std + 1e-10)
                
                sim0 = torch.dot(restored_normalized, p_0_normalized).item() / restored_flat.numel()
                sim1 = torch.dot(restored_normalized, p_1_normalized).item() / restored_flat.numel()
                max_sim = max(sim0, sim1)
                
                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_state = state
            
            states[i, j] = best_state
    
    # 找到第一个完整的水印块
    x0, y0 = xy0
    X0, Y0 = XY0
    first_block_x, first_block_y = (X0 - x0) // s, (Y0 - y0) // s
    
    # 推断所有完整的水印块，从最左上角网格位置开始
    watermark_blocks = infer_all_watermark_blocks(first_block_y, first_block_x, states, s, wm_h=wm_h, wm_w=wm_w, device=device)
    
    # 批量处理每个完整的水印块并提取水印信息
    watermarks = []
    for block_y, block_x, state in watermark_blocks:
        # 提取水印块并根据状态进行逆向变换
        block = image_t[:, block_y*s:(block_y+wm_h)*s, block_x*s:(block_x+wm_w)*s]
        restored_block = apply_inverse_transform(block, state, device=device)
        
        # 批量提取水印块内的所有比特
        wm_blocks = restored_block.unfold(1, s, s).unfold(2, s, s)  # [C, wm_h, wm_w, s, s]
        wm_blocks = wm_blocks.permute(1, 2, 0, 3, 4).reshape(wm_h * wm_w, -1)  # [wm_h*wm_w, C*s*s]
        
        # 批量计算所有比特与p_0, p_1的相似度
        wm_blocks_normalized = (wm_blocks - wm_blocks.mean(dim=1, keepdim=True)) / (wm_blocks.std(dim=1, keepdim=True) + 1e-10)
        sim0_batch = torch.matmul(wm_blocks_normalized, p_0_normalized) / wm_blocks.shape[1]  # [wm_h*wm_w]
        sim1_batch = torch.matmul(wm_blocks_normalized, p_1_normalized) / wm_blocks.shape[1]  # [wm_h*wm_w]
        
        watermark = (sim1_batch > sim0_batch).to(torch.int32).reshape(wm_h, wm_w)
        watermarks.append(watermark)
    
    # 批量处理图像周边的比特块
    vote_dict = {}
    
    # 创建水印块掩码以快速跳过
    in_watermark = torch.zeros((rows, cols), dtype=torch.bool, device=device)
    for wm_y, wm_x, _ in watermark_blocks:
        in_watermark[wm_y:wm_y+wm_h, wm_x:wm_x+wm_w] = True
    
    # 收集需要处理的边缘块
    edge_indices = []
    for i in range(rows):
        for j in range(cols):
            if not in_watermark[i, j]:
                edge_indices.append((i, j))
    
    # 批量处理边缘块
    if edge_indices:
        for i, j in edge_indices:
            block_3d = blocks[i, j]  # [C, s, s]
            state = int(states[i, j].item())
            restored_bit_block = apply_inverse_transform(block_3d, state, device=device)
            
            # 推断相对位置
            rel_i = (i - first_block_y) % wm_h
            rel_j = (j - first_block_x) % wm_w
            if state in [3, 4]:
                rel_i = wm_h - 1 - rel_i
            if state in [2, 4]:
                rel_j = wm_w - 1 - rel_j
            
            # 快速提取水印位
            restored_flat = restored_bit_block.flatten()
            restored_normalized = (restored_flat - restored_flat.mean()) / (restored_flat.std() + 1e-10)
            sim0 = torch.dot(restored_normalized, p_0_normalized).item() / restored_flat.numel()
            sim1 = torch.dot(restored_normalized, p_1_normalized).item() / restored_flat.numel()
            bit = 1 if sim1 > sim0 else 0
            
            # 记录投票
            key = (rel_i, rel_j)
            if key not in vote_dict:
                vote_dict[key] = []
            vote_dict[key].append(bit)
    
    # 合并水印信息和投票（保持在GPU上直到最后）
    final_watermark = torch.zeros((wm_h, wm_w), dtype=torch.int32, device=device)
    for i in range(wm_h):
        for j in range(wm_w):
            votes = []
            # 从完整水印块中获取
            for watermark in watermarks:
                votes.append(int(watermark[i, j].item()))
            # 从周边比特块中获取
            if (i, j) in vote_dict:
                votes.extend(vote_dict[(i, j)])
            # 投票
            final_watermark[i, j] = 1 if votes and sum(votes) > len(votes) / 2 else 0
    
    # 输出提取的水印信息
    # logINFO(f"提取的水印信息({wm_h}x{wm_w})：", f)
    # logINFO(final_watermark.cpu().numpy() if torch.is_tensor(final_watermark) else final_watermark, f)
    
    # 计算比特正确率
    if torch.is_tensor(true_watermark):
        true_wm_t = true_watermark.to(device)
    else:
        true_wm_t = torch.from_numpy(true_watermark).to(device)
    
    correct_bits = torch.sum(final_watermark == true_wm_t).item()
    total_bits = wm_h * wm_w
    accuracy = correct_bits / total_bits
    # logINFO(f"水印信息比特正确率: {accuracy * 100:.2f}%", f)

    return accuracy
