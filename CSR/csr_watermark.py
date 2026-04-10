"""
Camera Shooting Resilient (CSR) Watermarking Scheme for Underpainting Documents
Based on: Fang et al., IEEE TCSVT 2020

文字检测阈值应根据背景灰度动态调整，正确实现对角补偿方法以提高文字遮挡区域的提取准确率
"""

import numpy as np
import cv2
from scipy.fft import dct, idct
from typing import Tuple
import os


def dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block: np.ndarray) -> np.ndarray:
    """2D IDCT"""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


class CSRWatermark:
    """
    Camera Shooting Resilient Watermark System
    """
    

    def __init__(self, 
                 embedding_strength: float = 50.0,
                 block_size: int = 16,
                 watermark_rows: int = 16,
                 watermark_cols: int = 8,
                 underpainting_color: Tuple[int, int, int] = (199, 237, 204)):
        self.r = embedding_strength
        self.block_size = block_size
        self.a = watermark_rows
        self.b = watermark_cols
        self.c1_pos = (4, 5)
        self.c2_pos = (5, 4)
        self.underpainting_color = underpainting_color
        self.capacity = self.a * self.b
        self.unit_height = self.a * self.block_size
        self.unit_width = self.b * self.block_size
        
    def _embed_bit_in_block(self, block: np.ndarray, bit: int) -> np.ndarray:
        """Embed single bit using DCT coefficient exchange (Eq. 3)"""
        block_float = block.astype(np.float64)
        dct_block = dct2(block_float)
        
        if bit == 0:
            dct_block[self.c1_pos] = self.r
            dct_block[self.c2_pos] = -self.r
        else:
            dct_block[self.c1_pos] = -self.r
            dct_block[self.c2_pos] = self.r
        
        embedded_block = idct2(dct_block)
        return np.clip(embedded_block, 0, 255).astype(np.uint8)
    
    def _extract_bit_from_block(self, block: np.ndarray) -> int:
        """Extract single bit using DCT coefficient comparison (Eq. 24)"""
        if block.shape[0] != self.block_size or block.shape[1] != self.block_size:
            block = cv2.resize(block, (self.block_size, self.block_size), 
                              interpolation=cv2.INTER_LINEAR)
        
        block_float = block.astype(np.float64)
        dct_block = dct2(block_float)
        
        C1 = dct_block[self.c1_pos]
        C2 = dct_block[self.c2_pos]
        
        return 0 if C1 >= C2 else 1
    
    def embed(self, watermark_bits: np.ndarray, add_noise: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Embed watermark into underpainting"""
        bits = np.zeros(self.capacity, dtype=np.int8)
        length = min(len(watermark_bits), self.capacity)
        bits[:length] = watermark_bits[:length]
        
        W = bits.reshape((self.b, self.a)).T
        
        base_value = 180
        underpainting = np.ones((self.unit_height, self.unit_width), dtype=np.uint8) * base_value
        
        if add_noise:
            noise = np.random.normal(0, 2, underpainting.shape)
            underpainting = np.clip(underpainting.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        
        for i in range(self.a):
            for j in range(self.b):
                y1, y2 = i * self.block_size, (i + 1) * self.block_size
                x1, x2 = j * self.block_size, (j + 1) * self.block_size
                
                block = underpainting[y1:y2, x1:x2].copy()
                embedded = self._embed_bit_in_block(block, W[i, j])
                underpainting[y1:y2, x1:x2] = embedded
        
        return underpainting, W
    
    def extract(self, watermarked_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract watermark from underpainting image"""
        if len(watermarked_img.shape) == 3:
            gray = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = watermarked_img.copy()
        
        if gray.shape[0] != self.unit_height or gray.shape[1] != self.unit_width:
            gray = cv2.resize(gray, (self.unit_width, self.unit_height), 
                             interpolation=cv2.INTER_LINEAR)
        
        W = np.zeros((self.a, self.b), dtype=np.int8)
        
        for i in range(self.a):
            for j in range(self.b):
                y1, y2 = i * self.block_size, (i + 1) * self.block_size
                x1, x2 = j * self.block_size, (j + 1) * self.block_size
                
                block = gray[y1:y2, x1:x2]
                W[i, j] = self._extract_bit_from_block(block)
        
        bits = W.T.flatten()
        return bits, W
    
    def apply_flip(self, underpainting: np.ndarray) -> np.ndarray:
        """Apply flip arrangement (Eq. 4)"""
        P = underpainting
        P_h = np.fliplr(P)
        P_v = np.flipud(P)
        P_hv = np.flipud(np.fliplr(P))
        
        top = np.hstack([P, P_h])
        bottom = np.hstack([P_v, P_hv])
        return np.vstack([top, bottom])
    
    def extract_from_flipped(self, flipped_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract from flipped image"""
        h, w = flipped_img.shape[:2]
        quadrant = flipped_img[:h//2, :w//2]
        return self.extract(quadrant)
    
    def colorize(self, gray_img: np.ndarray) -> np.ndarray:
        """Convert grayscale to colored"""
        if len(gray_img.shape) == 3:
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        h, w = gray_img.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for c in range(3):
            colored[:, :, c] = np.clip(
                gray_img.astype(np.float64) * self.underpainting_color[c] / 255,
                0, 255
            ).astype(np.uint8)
        
        return colored
    
    def apply_to_document(self, 
                          document: np.ndarray,
                          watermark_bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply watermark to document"""
        h, w = document.shape[:2]
        
        underpainting, W = self.embed(watermark_bits, add_noise=False)
        flipped = self.apply_flip(underpainting)
        
        fh, fw = flipped.shape
        tiles_y = (h // fh) + 2
        tiles_x = (w // fw) + 2
        tiled = np.tile(flipped, (tiles_y, tiles_x))[:h, :w]
        
        colored_bg = self.colorize(tiled)
        
        if len(document.shape) == 3:
            doc_gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
        else:
            doc_gray = document.copy()
        
        _, text_mask = cv2.threshold(doc_gray, 200, 255, cv2.THRESH_BINARY)
        
        result = colored_bg.copy()
        text_pixels = text_mask == 0
        result[text_pixels] = [0, 0, 0]
        
        return result, underpainting, W
    
    def _diagonal_compensate(self, block: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
        """
        对角补偿 (Eq. 23)
        
        由于DCT系数(4,5)和(5,4)关于主对角线对称，嵌入后的块具有对角对称特性。
        被文字遮挡的像素可用其对角位置的像素替代。
        
        D(x,y) = D(N-1-x, N-1-y), 如果当前位置是文字且对角位置不是文字
               = E(D),            否则
        """
        h, w = block.shape
        compensated = block.astype(np.float64).copy()
        
        # 计算背景像素的均值
        bg_pixels = block[text_mask >= 128]
        bg_mean = np.mean(bg_pixels) if len(bg_pixels) > 0 else np.mean(block)
        
        for y in range(h):
            for x in range(w):
                if text_mask[y, x] < 128:  # 当前位置是文字（黑色）
                    dy, dx = h - 1 - y, w - 1 - x
                    if text_mask[dy, dx] >= 128:  # 对角位置是背景
                        compensated[y, x] = block[dy, dx]
                    else:
                        compensated[y, x] = bg_mean
        
        return compensated.astype(np.uint8)
    
    def extract_from_document(self, 
                              document: np.ndarray,
                              avoid_text: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        从文档中提取水印，使用对角补偿处理文字区域
        
        论文 Section IV-E, Figure 14, Eq. 23
        """
        if len(document.shape) == 3:
            gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
        else:
            gray = document.copy()
        
        h, w = gray.shape
        fh = self.unit_height * 2
        fw = self.unit_width * 2
        
        # 取第一个完整的翻转单元
        region = gray[:fh, :fw]
        watermark_region = region[:fh//2, :fw//2]
        
        if not avoid_text:
            return self.extract(watermark_region)
        
        # 关键：动态计算文字检测阈值
        # 背景是彩色底纹（灰度约140-160），文字是黑色（0）
        # 使用阈值 = 背景均值的一半
        bg_estimate = np.percentile(watermark_region, 75)  # 取75%分位数作为背景估计
        text_threshold = bg_estimate / 2
        
        _, text_mask = cv2.threshold(watermark_region, text_threshold, 255, cv2.THRESH_BINARY)
        
        W = np.zeros((self.a, self.b), dtype=np.int8)
        
        for i in range(self.a):
            for j in range(self.b):
                y1, y2 = i * self.block_size, (i + 1) * self.block_size
                x1, x2 = j * self.block_size, (j + 1) * self.block_size
                
                block = watermark_region[y1:y2, x1:x2]
                mask = text_mask[y1:y2, x1:x2]
                
                text_ratio = np.sum(mask < 128) / mask.size
                
                if text_ratio > 0.05:  # 超过5%是文字，需要补偿
                    compensated = self._diagonal_compensate(block, mask)
                    W[i, j] = self._extract_bit_from_block(compensated)
                else:
                    W[i, j] = self._extract_bit_from_block(block)
        
        bits = W.T.flatten()
        return bits, W


def message_to_bits(message: str) -> np.ndarray:
    """Convert string to binary array"""
    bits = []
    for char in message:
        bits.extend([int(b) for b in format(ord(char), '08b')])
    return np.array(bits, dtype=np.int8)


def bits_to_message(bits: np.ndarray) -> str:
    """Convert binary array to string"""
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = bits[i:i+8]
        char_code = int(''.join(map(str, byte)), 2)
        if 32 <= char_code <= 126:
            chars.append(chr(char_code))
    return ''.join(chars)


def compute_ber(original: np.ndarray, extracted: np.ndarray) -> Tuple[float, int]:
    """Compute Bit Error Rate"""
    min_len = min(len(original), len(extracted))
    if min_len == 0:
        return 1.0, min_len
    errors = np.sum(original[:min_len] != extracted[:min_len])
    return errors / min_len, int(errors)


def run_tests():
    """运行完整测试"""
    print("=" * 70)
    print("CSR Watermark - 完整测试")
    print("=" * 70)
    

    csrw = CSRWatermark(embedding_strength=30.0)
    
    message = "3354421163338888"
    original_bits = message_to_bits(message)
    original_bits = np.concatenate([
        original_bits, 
        np.zeros(128 - len(original_bits), dtype=np.int8)
    ])
    
    print(f"\n测试消息: '{message}'")
    print("-" * 70)
    
    # 测试1: 纯底纹
    print("\n[测试1] 纯底纹")
    watermarked, _ = csrw.embed(original_bits)
    cv2.imwrite("CSR/watermarked.png", watermarked)
    extracted, _ = csrw.extract(watermarked)
    ber, errors = compute_ber(original_bits, extracted)
    msg = bits_to_message(extracted)
    status = "✓" if errors == 0 else "✗"
    print(f"  提取消息: '{msg}', 错误: {errors}/128 {status}")
    
    # 测试2: 翻转排列
    print("\n[测试2] 翻转排列")
    flipped = csrw.apply_flip(watermarked)
    cv2.imwrite("CSR/watermarked_flipped.png", flipped)
    extracted, _ = csrw.extract_from_flipped(flipped)
    ber, errors = compute_ber(original_bits, extracted)
    msg = bits_to_message(extracted)
    status = "✓" if errors == 0 else "✗"
    print(f"  提取消息: '{msg}', 错误: {errors}/128 {status}")
    
    # 测试3: 彩色底纹
    print("\n[测试3] 彩色底纹")
    colored = csrw.colorize(flipped)
    cv2.imwrite("CSR/colored.png", colored)
    extracted, _ = csrw.extract_from_flipped(colored)
    ber, errors = compute_ber(original_bits, extracted)
    msg = bits_to_message(extracted)
    status = "✓" if errors == 0 else "✗"
    print(f"  提取消息: '{msg}', 错误: {errors}/128 {status}")
    
    # 测试4: 文档（使用对角补偿）
    print("\n[测试4] 带文字文档（对角补偿）")
    doc_path = "CSR/document.png"
    if os.path.exists(doc_path):
        document = cv2.imread(doc_path)
        print(f"  文档尺寸: {document.shape}")
        
        watermarked_doc, _, _ = csrw.apply_to_document(document, original_bits)
        cv2.imwrite("CSR/watermarked_document.png", watermarked_doc)
        
        loaded_doc = cv2.imread("CSR/watermarked_document.png")
        extracted, _ = csrw.extract_from_document(loaded_doc, avoid_text=True)
        ber, errors = compute_ber(original_bits, extracted)
        msg = bits_to_message(extracted)
        status = "✓" if errors <= 10 else "✗"
        print(f"  提取消息: '{msg}', 错误: {errors}/128 {status}, Accuracy: {100 - (errors / 128 * 100):.2f}%")
    else:
        print(f"  文档不存在: {doc_path}")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()