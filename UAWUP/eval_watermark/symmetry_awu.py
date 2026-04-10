import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter, generic_filter
from Tools.Baseimagetool import *
from Tools.ImageIO import *

def calculate_autoconvolution(image):
    """
    计算图像的自卷积和（不归一化）。

    参数：
    - image: 输入的彩色图像 (H, W, 3)，范围 0-255

    返回：
    - autoconvolution: 自卷积和 (H, W)
    """
    image = image.astype(np.float32) / 255.0  # 归一化到 0-1
    h, w, channels = image.shape
    autoconvolution = np.zeros((h, w), dtype=np.float32)
    
    for c in range(channels):
        channel = image[:, :, c]
        fft_image = fft2(channel)
        autoconvolution_channel = ifft2(fft_image * fft_image).real
        autoconvolution += autoconvolution_channel
    
    return autoconvolution

def normalize_adjusted_symmetry(symmetry_adjusted):
    """
    对调整后的对称性图像应用归一化（symmetry_adjusted / denominator）。

    参数：
    - symmetry_adjusted: 调整后的对称性图 (H, W)

    返回：
    - symmetry_normalized: 归一化后的对称性图 (H, W)
    """
    ones = np.ones_like(symmetry_adjusted, dtype=np.float32)
    fft_ones = fft2(ones)
    denominator = ifft2(fft_ones * fft_ones).real + 1e-10
    symmetry_normalized = symmetry_adjusted / denominator
    return symmetry_normalized

def save_symmetry_image(symmetry, output_path):
    """
    保存对称性图像。

    参数：
    - symmetry: 对称性图 (H, W)
    - output_path: 保存路径
    """
    symmetry_normalized = (symmetry - np.min(symmetry)) / (np.max(symmetry) - np.min(symmetry) + 1e-10)
    symmetry_image = (symmetry_normalized * 255).astype(np.uint8)
    cv2.imwrite(output_path, symmetry_image)

def calculate_and_save_symmetry_signals(symmetry_normalized, save_path, f):
    """
    计算并保存行对称性和列对称性信号图，并检测峰值点。
    
    参数：
    - symmetry_normalized: 归一化后的对称性图 (H, W)
    - save_path: 保存路径的前缀
    - f: 日志函数句柄
    
    返回：
    - column_symmetry: 列对称性信号，一种用于底纹文档的相机拍摄可恢复水印方案中定义的列为对称轴
    - row_symmetry: 行对称性信号，一种用于底纹文档的相机拍摄可恢复水印方案中定义的行为对称轴
    """    
    # 计算列对称性（沿着列方向累加）
    column_symmetry = np.sum(symmetry_normalized, axis=0)
    # 计算行对称性（沿着行方向累加）
    row_symmetry = np.sum(symmetry_normalized, axis=1)
    
    # 归一化列对称性信号
    if np.max(column_symmetry) != np.min(column_symmetry):
        column_symmetry_norm = (column_symmetry - np.min(column_symmetry)) / (np.max(column_symmetry) - np.min(column_symmetry))
    else:
        column_symmetry_norm = np.zeros_like(column_symmetry)
    
    # 归一化行对称性信号
    if np.max(row_symmetry) != np.min(row_symmetry):
        row_symmetry_norm = (row_symmetry - np.min(row_symmetry)) / (np.max(row_symmetry) - np.min(row_symmetry))
    else:
        row_symmetry_norm = np.zeros_like(row_symmetry)
    
    # 寻找列对称性信号的峰值点
    column_peaks, _ = find_peaks(column_symmetry_norm, height=0.5, distance=30)
    
    # 寻找行对称性信号的峰值点
    row_peaks, _ = find_peaks(row_symmetry_norm, height=0.5, distance=30)
    
    # 绘制列对称性信号图
    plt.figure(figsize=(12, 6))
    plt.plot(column_symmetry_norm)
    plt.title('Column Symmetry Signal')
    plt.xlabel('Column Index')
    plt.ylabel('Normalized Symmetry Value')
    plt.grid(True)
    
    # 标记峰值点
    plt.plot(column_peaks, column_symmetry_norm[column_peaks], 'ro')
    for peak in column_peaks:
        plt.text(peak, column_symmetry_norm[peak] + 0.02, f'({peak}, {column_symmetry_norm[peak]:.2f})', 
                 ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path + '_column_symmetry.png', dpi=150)
    plt.close()
    
    # 绘制行对称性信号图
    plt.figure(figsize=(12, 6))
    plt.plot(row_symmetry_norm)
    plt.title('Row Symmetry Signal')
    plt.xlabel('Row Index')
    plt.ylabel('Normalized Symmetry Value')
    plt.grid(True)
    
    # 标记峰值点
    plt.plot(row_peaks, row_symmetry_norm[row_peaks], 'ro')
    for peak in row_peaks:
        plt.text(peak, row_symmetry_norm[peak] + 0.02, f'({peak}, {row_symmetry_norm[peak]:.2f})', 
                 ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path + '_row_symmetry.png', dpi=150)
    plt.close()
    
    # 记录峰值点
    logINFO(f"列对称性峰值点: {list(zip(column_peaks, column_symmetry_norm[column_peaks]))}", f)
    logINFO(f"行对称性峰值点: {list(zip(row_peaks, row_symmetry_norm[row_peaks]))}", f)
    
    return column_symmetry, row_symmetry

def symmetry_awu(window_size, save_path, f):
    """
    主函数：读取图像，计算对称性，减去文本对称性，归一化并检测角点。
    """
    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取对抗水印文档图像
    aw_image = img_read(save_path + '.png')
    
    # 计算对抗水印文档图像的自卷积（不归一化）
    symmetry_aw = calculate_autoconvolution(aw_image)
    
    # 保存对抗水印文档图像的自卷积图像
    # save_symmetry_image(symmetry_aw, save_path + '_awti_symmetry.png')
    # logINFO(f"对抗水印文档图像自卷积已保存至 {save_path + '_awti_symmetry.png'}")
    
    # 对调整后的自卷积进行归一化
    symmetry_normalized = normalize_adjusted_symmetry(symmetry_aw)
    
    # 保存归一化后的对称性图像
    save_symmetry_image(symmetry_normalized, save_path + '_symmetry_normalized.png')
    # logINFO(f"归一化对称性图像已保存至 {save_path + '_symmetry_normalized.png'}")
       
    # 计算并保存行列对称性信号图
    column_symmetry, row_symmetry = calculate_and_save_symmetry_signals(symmetry_normalized, save_path, f)
    logINFO(f"行列对称性信号图已保存至 {save_path + '_row_symmetry.png'} 和 {save_path + '_column_symmetry.png'}", f)


if __name__ == "__main__":
    s = 30
    save_path = 'eval_watermark/awu/random_awu'
    log_path = os.path.join(os.path.dirname(save_path), "log.txt")
    log_file = open(log_path, 'w')
    symmetry_awu(s, save_path, log_file)
