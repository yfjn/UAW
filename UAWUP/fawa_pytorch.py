#!/usr/bin/env python3
"""
Paragraph Adversarial Watermark Attack on STR (PyTorch Version)
https://github.com/strongman1995/Fast-Adversarial-Watermark-Attack-on-OCR

功能：
1. 使用 ImageDataset 生成文本图像
2. 添加显式水印（类似ECML PKDD风格）
3. 在水印覆盖处执行对抗攻击（攻击STR，CTC损失）
4. 统计单词攻击成功率
5. 输出结果图像和日志

Usage:
    python fawa_pytorch.py [--num_images 10] [--output_dir results]
"""

import os
import sys
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/UAW/")
import argparse
import logging
import re
from datetime import datetime
from typing import List, Tuple, Dict
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from UAWUP.dataset import ImageDataset
from torchvision import transforms

# NLTK for corpus
import nltk
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown', quiet=True)
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)
from nltk.corpus import brown, words as nltk_words

# EasyOCR for STR
import easyocr
from Tools.ImageIO import tensor_to_pil

# tqdm for progress bar
from tqdm import tqdm


# ============================================================
# 日志设置
# ============================================================
def setup_logger(output_dir: str) -> Tuple[logging.Logger, str]:
    """设置日志器，同时输出到终端和文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger('ParagraphAttack')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 终端输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件输出
    log_file = os.path.join(output_dir, f'attack_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger, log_file


# ============================================================
# 水印生成与添加
# ============================================================
class WatermarkGenerator:
    """显式水印生成器"""
    
    def __init__(
        self,
        text: str = 'TDSC',
        color: Tuple[int, int, int] = (200, 150, 100),
        size: int = 60,  # 水印字体大小
        angle: int = 10,
    ):
        self.text = text
        self.color = color
        self.size = size
        self.angle = angle
        self.watermark_img = self._generate_watermark()
    
    def _generate_watermark(self) -> Image.Image:
        """生成水印图像"""
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        ]
        
        font = None
        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, self.size)
                    break
                except:
                    continue
        
        if font is None:
            font = ImageFont.load_default()
        
        # 计算文本尺寸
        dummy = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy)
        try:
            bbox = draw.textbbox((0, 0), self.text, font=font)
            text_width = bbox[2] - bbox[0] + 20
            text_height = bbox[3] - bbox[1] + 20
        except:
            text_width = len(self.text) * self.size
            text_height = self.size + 20
        
        # 创建水印图像
        wm_img = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(wm_img)
        draw.text((10, 5), self.text, font=font, fill=self.color + (255,))
        
        # 旋转
        wm_img = wm_img.rotate(self.angle, expand=True, fillcolor=(255, 255, 255, 0))
        
        return wm_img
    
    def add_watermark_to_image(
        self,
        image: Image.Image,
        spacing_x: int = 0,  # 水印之间的水平间距（像素）
        spacing_y: int = 0,  # 水印之间的垂直间距（像素）
    ) -> Tuple[Image.Image, np.ndarray, List[Tuple[int, int]]]:
        """
        在图像上均匀铺设水印
        
        Returns:
            watermarked_image: 添加水印后的图像 (PIL Image)
            watermark_mask: 水印掩码 (H, W)，True 表示水印区域
            positions: 水印位置列表
        """
        if image.mode == 'L':
            image = image.convert('RGB')
        
        img_w, img_h = image.size
        wm_w, wm_h = self.watermark_img.size
        
        # 创建水印图层
        wm_layer = Image.new('RGBA', (img_w, img_h), (255, 255, 255, 0))
        positions = []
        
        y = -wm_h // 2
        row = 0
        while y < img_h:
            x_offset = (wm_w // 2 + spacing_x // 2) if row % 2 == 1 else 0
            x = -x_offset
            while x < img_w:
                if x + wm_w > 0 and x < img_w and y + wm_h > 0:
                    wm_layer.paste(self.watermark_img, (x, y), self.watermark_img)
                    positions.append((x, y))
                x += wm_w + spacing_x
            y += wm_h + spacing_y
            row += 1
        
        # 创建水印掩码（水印非透明区域）
        wm_arr = np.array(wm_layer)
        wm_mask = wm_arr[:, :, 3] > 50  # Alpha通道大于50的区域
        
        # 合成
        result = image.copy().convert('RGBA')
        result = Image.alpha_composite(result, wm_layer)
        
        # 保留原始文字（文字区域不被水印覆盖）
        orig_gray = np.array(image.convert('L'))
        text_mask = orig_gray < 200  # 文字区域
        
        result_rgb = result.convert('RGB')
        result_arr = np.array(result_rgb)
        orig_arr = np.array(image.convert('RGB'))
        
        # 文字区域使用原始图像
        for c in range(3):
            result_arr[:, :, c] = np.where(text_mask, orig_arr[:, :, c], result_arr[:, :, c])
        
        return Image.fromarray(result_arr), wm_mask, positions


# ============================================================
# 单词替换工具
# ============================================================
class WordReplacer:
    """单词替换工具"""
    
    def __init__(self):
        # 放宽单词长度限制：3-10个字母
        self.word_list = set(w.lower() for w in nltk_words.words() if w.isalpha() and 3 <= len(w) <= 10)
    
    def find_similar_word(self, word: str) -> str:
        """找到与给定单词长度相同、最相似的不同单词"""
        word_len = len(word)
        candidates = [w for w in self.word_list if len(w) == word_len and w != word.lower()]
        
        if not candidates:
            return word
        
        word_lower = word.lower()
        
        def similarity(w):
            return sum(c1 == c2 for c1, c2 in zip(word_lower, w))
        
        scored = [(w, similarity(w)) for w in candidates]
        scored.sort(key=lambda x: -x[1])
        
        # 选择最相似的单词
        best = scored[0][0]
        
        if word.istitle():
            best = best.capitalize()
        elif word.isupper():
            best = best.upper()
        
        return best
    
    def get_word_positions(self, text: str, line_x_start: int, line_x_end: int) -> List[Tuple[str, int, int, int]]:
        """
        获取每个单词的像素位置（基于行的实际边界框）
        
        Args:
            text: 文本
            line_x_start: 行的起始x坐标
            line_x_end: 行的结束x坐标
        
        Returns:
            [(word, idx, start_x, end_x), ...]
        """
        words = text.split(' ')
        if not words or not text.strip():
            return []
        
        # 使用行的实际宽度来计算字符宽度
        line_width = line_x_end - line_x_start
        total_chars = len(text)
        if total_chars == 0:
            return []
        
        char_width = line_width / total_chars
        
        positions = []
        current_x = line_x_start  # 从行的起始位置开始
        
        for idx, word in enumerate(words):
            start_x = int(current_x)
            end_x = int(current_x + len(word) * char_width)
            
            positions.append((word, idx, start_x, end_x))
            
            # 移动到下一个单词（加上空格）
            current_x += (len(word) + 1) * char_width
        
        return positions
    
    def find_covered_words(
        self,
        text: str,
        wm_mask: np.ndarray,
        line_y_start: int,
        line_y_end: int,
        line_x_start: int,
        line_x_end: int,
        coverage_threshold: float = 0.3,  # 30%覆盖阈值
    ) -> List[Dict]:
        """
        找到被水印覆盖的单词
        
        判定方式：像素覆盖比例（同时考虑x和y方向）
        - 计算单词所在矩形区域中被水印覆盖的像素比例
        - 如果比例 >= coverage_threshold，则认为该单词被水印覆盖
        """
        if len(wm_mask.shape) != 2:
            return []
        
        img_height, img_width = wm_mask.shape
        
        # 确保y坐标在有效范围内
        line_y_start = max(0, min(line_y_start, img_height - 1))
        line_y_end = max(0, min(line_y_end, img_height))
        
        if line_y_start >= line_y_end:
            return []
        
        # 使用行的实际边界框计算单词位置
        positions = self.get_word_positions(text, line_x_start, line_x_end)
        covered = []
        
        for word, idx, start_x, end_x in positions:
            # 确保x坐标在掩码范围内
            start_x = max(0, min(start_x, img_width - 1))
            end_x = max(0, min(end_x, img_width))
            
            if end_x <= start_x:
                continue
            
            # 提取单词区域的掩码（同时考虑x和y方向）
            region = wm_mask[line_y_start:line_y_end, start_x:end_x]
            
            if region.size > 0:
                # 像素覆盖比例判定
                coverage = np.sum(region) / region.size
                
                clean_word = re.sub(r'[^a-zA-Z]', '', word)
                # 放宽单词长度限制：3-10个字母
                if coverage >= coverage_threshold and 3 <= len(clean_word) <= 10:
                    covered.append({
                        'word': word,
                        'word_idx': idx,
                        'coverage': coverage,
                        'clean_word': clean_word,
                    })
        
        return covered
    
    def replace_words_in_text(self, text: str, replacements: List[Dict]) -> Tuple[str, List[Dict]]:
        """替换文本中的单词"""
        words = text.split(' ')
        replacement_info = []
        
        for repl in replacements:
            idx = repl['word_idx']
            original = repl['word']
            clean = repl['clean_word']
            
            new_word = self.find_similar_word(clean)
            
            if new_word != clean.lower() and new_word != clean:
                if original and original[-1] in '.,;:!?':
                    new_word = new_word + original[-1]
                
                if idx < len(words):
                    words[idx] = new_word
                    replacement_info.append({
                        'word_idx': idx,
                        'original': original,
                        'target': new_word,
                        'coverage': repl['coverage'],
                    })
        
        return ' '.join(words), replacement_info


# ============================================================
# EasyOCR 包装器（用于STR攻击）
# ============================================================
class EasyOCRWrapper:
    """EasyOCR 包装器，用于文本识别"""
    
    def __init__(self, device: str = 'cpu'):
        self.device_str = device
        gpu = device != 'cpu' and torch.cuda.is_available()
        self.reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
        
        # 获取内部模型和字符集
        self.recognizer = self.reader.recognizer
        self.character = self.reader.character
        
        # 创建字符到索引的映射
        # EasyOCR的character列表中，index 0通常是blank '[blank]'
        self.char_to_idx = {char: idx for idx, char in enumerate(self.character)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.character)}
        
        # 打印字符集信息
        print(f"Character set size: {len(self.character)}")
        print(f"First 10 chars: {self.character[:10]}")
        
        # 查找blank index
        # EasyOCR通常使用0作为blank，或者字符集末尾
        # 打印字符集信息用于调试
        print(f"Character set size: {len(self.character)}")
        print(f"First 10 chars: {self.character[:10]}")
        print(f"Last 5 chars: {self.character[-5:]}")
        
        # EasyOCR的blank通常在index 0，对应的字符可能是空字符串或特殊标记
        self.blank_idx = 0
        
        # 查找可能的blank标记
        for i, c in enumerate(self.character):
            if c in ['[blank]', '[s]', '[GO]', '', ' ']:
                print(f"Found potential blank at index {i}: '{c}'")
        
        print(f"Using blank index: {self.blank_idx}")
    
    def recognize(self, image: Image.Image) -> str:
        """识别图像中的文本"""
        img_arr = np.array(image.convert('RGB'))
        results = self.reader.readtext(img_arr)
        texts = [result[1] for result in results]
        return ' '.join(texts)
    
    def recognize_with_boxes(self, image: Image.Image) -> List[Tuple]:
        """识别图像并返回边界框"""
        img_arr = np.array(image.convert('RGB'))
        results = self.reader.readtext(img_arr)
        return results
    
    def text_to_labels(self, text: str) -> Tuple[torch.Tensor, int]:
        """将文本转换为标签张量（不包含blank）"""
        labels = []
        
        for char in text:
            idx = None
            # 尝试原字符
            if char in self.char_to_idx:
                idx = self.char_to_idx[char]
            # 尝试小写
            elif char.lower() in self.char_to_idx:
                idx = self.char_to_idx[char.lower()]
            # 尝试大写
            elif char.upper() in self.char_to_idx:
                idx = self.char_to_idx[char.upper()]
            
            # 跳过blank token和未知字符
            if idx is not None and idx != self.blank_idx:
                labels.append(idx)
        
        if not labels:
            # 如果没有有效标签，至少返回一个非blank字符
            for i in range(len(self.character)):
                if i != self.blank_idx:
                    labels = [i]
                    break
        
        return torch.tensor(labels, dtype=torch.long), len(labels)


# ============================================================
# 对抗攻击器（基于CTC损失的白盒攻击）
# ============================================================
class AdversarialAttacker:
    """对抗水印攻击器（攻击STR，使用CTC损失）"""
    
    def __init__(
        self,
        ocr_reader: EasyOCRWrapper,
        device: torch.device,
        eps: float = 0.2,
        eps_iter: float = 5 / 255,
        nb_iter: int = 1000,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        decay: float = 1.0,
    ):
        self.ocr_reader = ocr_reader
        self.device = device
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.decay = decay
        self.blank_idx = ocr_reader.blank_idx
        
        # 获取EasyOCR的识别模型
        self.recognizer = ocr_reader.recognizer
        self.num_classes = len(ocr_reader.character)
        
        if self.recognizer is not None:
            # 如果是DataParallel，获取内部模块
            if isinstance(self.recognizer, nn.DataParallel):
                self.recognizer = self.recognizer.module
            
            self.recognizer = self.recognizer.to(device)
            # 设置为训练模式
            self.recognizer.train()
            # 冻结参数
            for param in self.recognizer.parameters():
                param.requires_grad = False
            
            # 禁用cudnn优化，避免backward问题
            torch.backends.cudnn.enabled = False
    
    def _preprocess_crop(self, crop_tensor: torch.Tensor) -> torch.Tensor:
        """
        预处理裁剪区域用于OCR识别
        
        Args:
            crop_tensor: (1, 3, H, W) RGB图像张量，值域[0, 1]
        
        Returns:
            (1, 1, 64, W') 灰度图张量，值域[-1, 1]
        """
        # RGB转灰度
        gray = 0.299 * crop_tensor[:, 0:1, :, :] + \
               0.587 * crop_tensor[:, 1:2, :, :] + \
               0.114 * crop_tensor[:, 2:3, :, :]
        
        _, _, h, w = gray.shape
        
        # 调整到高度64，保持宽高比
        new_h = 64
        new_w = max(int(w * new_h / h), 32)  # 最小宽度32
        
        gray = F.interpolate(gray, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # 归一化到[-1, 1]
        gray = (gray - 0.5) / 0.5
        
        return gray
    
    def _is_similar(self, s1: str, s2: str, threshold: float = 0.5) -> bool:
        """判断两个字符串是否相似"""
        if not s1 or not s2:
            return False
        if abs(len(s1) - len(s2)) > 2:
            return False
        matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        return matches / max(len(s1), len(s2)) >= threshold
    
    def attack_image(
        self,
        wm_image: Image.Image,
        wm_mask: np.ndarray,
        target_words: List[Dict],
        original_text: str,
        target_text: str,
        logger: logging.Logger,
    ) -> Tuple[Image.Image, int, float, float, str]:
        """
        对带水印图像执行对抗攻击
        
        策略：对每个目标单词区域分别计算CTC损失
        """
        # 转换图像为张量
        img_arr = np.array(wm_image.convert('RGB')).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_orig = img_tensor.clone()
        
        _, _, img_h, img_w = img_tensor.shape
        
        # 水印掩码
        mask_tensor = torch.from_numpy(wm_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        if mask_tensor.shape[1] == 1:
            mask_tensor = mask_tensor.repeat(1, 3, 1, 1)
        if mask_tensor.shape[2:] != img_tensor.shape[2:]:
            mask_tensor = F.interpolate(mask_tensor, size=img_tensor.shape[2:], mode='nearest')
        
        # 初始化
        perturbation = torch.zeros_like(img_tensor)
        momentum = torch.zeros_like(img_tensor)
        
        best_adv = img_tensor.clone()
        best_success = 0
        best_score = -float('inf')
        iterations = 0
        
        target_words_set = set(re.sub(r'[^a-zA-Z]', '', w['target']).lower() for w in target_words)
        original_words_set = set(re.sub(r'[^a-zA-Z]', '', w['original']).lower() for w in target_words)
        total_targets = len(target_words)
        
        # 获取每个目标单词的区域
        word_regions = []
        detection_results = self.ocr_reader.recognize_with_boxes(wm_image)
        
        for word_info in target_words:
            original_word = word_info['original']
            target_word = word_info['target']
            orig_clean = re.sub(r'[^a-zA-Z]', '', original_word).lower()
            
            for bbox, text, conf in detection_results:
                text_clean = re.sub(r'[^a-zA-Z]', '', text).lower()
                
                if text_clean == orig_clean or self._is_similar(text_clean, orig_clean, 0.5):
                    pts = np.array(bbox)
                    x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
                    y_min, y_max = int(pts[:, 1].min()), int(pts[:, 1].max())
                    
                    padding = 5
                    x_min = max(0, x_min - padding)
                    x_max = min(img_w, x_max + padding)
                    y_min = max(0, y_min - padding)
                    y_max = min(img_h, y_max + padding)
                    
                    target_labels, target_len = self.ocr_reader.text_to_labels(target_word)
                    
                    word_regions.append({
                        'original': original_word,
                        'target': target_word,
                        'bbox': (x_min, y_min, x_max, y_max),
                        'target_labels': target_labels.to(self.device),
                        'target_len': target_len
                    })
                    break
        
        logger.info(f"Found {len(word_regions)} word regions")
        for region in word_regions:
            bbox = region['bbox']
            logger.info(f"  '{region['original']}' -> '{region['target']}' bbox={bbox} target_len={region['target_len']}")
        
        # 测试recognizer
        test_success = False
        if self.recognizer is not None and len(word_regions) > 0:
            try:
                region = word_regions[0]
                x_min, y_min, x_max, y_max = region['bbox']
                test_crop = img_tensor[:, :, y_min:y_max, x_min:x_max]
                test_input = self._preprocess_crop(test_crop)
                
                logger.info(f"Test crop: {test_crop.shape}, preprocessed: {test_input.shape}")
                
                with torch.no_grad():
                    test_output = self.recognizer(test_input, None)
                
                seq_len = test_output.shape[1]
                logger.info(f"Output shape: {test_output.shape}, seq_len={seq_len}, target_len={region['target_len']}")
                
                if seq_len >= region['target_len']:
                    test_success = True
                    logger.info("CTC constraint OK")
                    
            except Exception as e:
                logger.info(f"Recognizer test failed: {e}")
        
        if not test_success:
            logger.info("Using black-box gradient estimation")
        
        pbar = tqdm(range(self.nb_iter), desc="Attacking", leave=False)
        
        for i in pbar:
            # 当前对抗图像
            adv_tensor = (img_orig + perturbation).clamp(self.clip_min, self.clip_max)
            adv_tensor = adv_tensor.detach().requires_grad_(True)
            
            total_loss = torch.tensor(0.0, device=self.device)
            loss_count = 0
            
            # 对每个单词区域计算CTC损失
            if test_success and self.recognizer is not None:
                # 确保模型在训练模式（否则cudnn RNN无法反向传播）
                self.recognizer.train()
                
                for region in word_regions:
                    try:
                        x_min, y_min, x_max, y_max = region['bbox']
                        target_labels = region['target_labels']
                        target_len = region['target_len']
                        
                        # 裁剪区域（保持梯度）
                        crop_tensor = adv_tensor[:, :, y_min:y_max, x_min:x_max]
                        
                        if crop_tensor.numel() == 0:
                            continue
                        
                        # 预处理（保持梯度链）
                        preprocessed = self._preprocess_crop(crop_tensor)
                        
                        # 前向传播
                        preds = self.recognizer(preprocessed, None)
                        seq_len = preds.size(1)
                        
                        if seq_len < target_len:
                            continue
                        
                        # CTC损失
                        log_probs = F.log_softmax(preds, dim=2).permute(1, 0, 2)
                        input_lengths = torch.tensor([seq_len], device=self.device)
                        target_lengths = torch.tensor([target_len], device=self.device)
                        
                        ctc_loss = F.ctc_loss(
                            log_probs,
                            target_labels.unsqueeze(0),
                            input_lengths,
                            target_lengths,
                            blank=self.blank_idx,
                            reduction='mean',
                            zero_infinity=True
                        )
                        
                        if i == 0:
                            logger.info(f"  '{region['original']}' loss: {ctc_loss.item():.2f}")
                        
                        if not torch.isnan(ctc_loss) and not torch.isinf(ctc_loss) and ctc_loss.item() > 0:
                            total_loss = total_loss + ctc_loss
                            loss_count += 1
                            
                    except Exception as e:
                        if i == 0:
                            logger.info(f"CTC error: {e}")
                        continue
            
            # 计算梯度
            if loss_count > 0:
                avg_loss = total_loss / loss_count
                avg_loss.backward()
                
                if adv_tensor.grad is not None:
                    grad = adv_tensor.grad.clone()
                    loss_val = avg_loss.item()
                    
                    if i == 0:
                        logger.info(f"Total loss: {loss_val:.2f}, regions: {loss_count}")
                else:
                    grad = self._estimate_gradient_fast(
                        adv_tensor.detach(), mask_tensor, target_words_set, original_words_set
                    )
                    loss_val = 0.0
            else:
                grad = self._estimate_gradient_fast(
                    (img_orig + perturbation).clamp(self.clip_min, self.clip_max),
                    mask_tensor,
                    target_words_set,
                    original_words_set
                )
                loss_val = 0.0
            
            # 梯度归一化 (类似TF版本的处理)
            grad_abs_mean = torch.mean(torch.abs(grad))
            if grad_abs_mean > 1e-12:
                grad_norm = grad / grad_abs_mean
            else:
                grad_norm = torch.sign(torch.randn_like(grad)) * mask_tensor
            
            # 动量累积
            momentum = self.decay * momentum + grad_norm
            
            # 应用mask
            grad_masked = momentum * mask_tensor
            
            # 更新扰动 (梯度下降，最小化CTC损失)
            perturbation = perturbation - self.eps_iter * torch.sign(grad_masked)
            perturbation = torch.clamp(perturbation, -self.eps, self.eps)
            
            # 评估当前结果
            current_adv = (img_orig + perturbation).clamp(self.clip_min, self.clip_max)
            current_pil = tensor_to_pil(current_adv)
            current_text = self.ocr_reader.recognize(current_pil)
            
            current_words = set(re.sub(r'[^a-zA-Z\s]', '', current_text).lower().split())
            success_count = len(target_words_set.intersection(current_words))
            orig_remaining = len(original_words_set.intersection(current_words))
            current_score = success_count - 0.3 * orig_remaining
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss_val:.2f}',
                'success': f'{success_count}/{total_targets}'
            })
            
            # 更新最佳结果
            if success_count > best_success or (success_count == best_success and current_score > best_score):
                best_success = success_count
                best_score = current_score
                best_adv = current_adv.clone().detach()
                iterations = i + 1
            
            # 提前停止
            if success_count >= total_targets:
                logger.info(f"Attack succeeded at iteration {i+1}")
                break
        
        pbar.close()
        
        # 计算扰动范数
        final_pert = (best_adv - img_orig).detach().cpu().numpy().squeeze()
        l2_norm = np.sqrt(np.sum(final_pert ** 2))
        linf_norm = np.max(np.abs(final_pert))
        
        adv_image = tensor_to_pil(best_adv)
        adv_text = self.ocr_reader.recognize(adv_image)
        
        return adv_image, iterations, l2_norm, linf_norm, adv_text
    
    def _estimate_gradient_fast(
        self, 
        img_tensor: torch.Tensor, 
        mask_tensor: torch.Tensor,
        target_words_set: set,
        original_words_set: set,
        num_samples: int = 30,
        delta: float = 0.1,
    ) -> torch.Tensor:
        """改进的黑盒梯度估计"""
        grad = torch.zeros_like(img_tensor)
        
        for _ in range(num_samples):
            noise = torch.randn_like(img_tensor) * delta
            noise = noise * mask_tensor
            
            # 正向扰动
            adv_pos = torch.clamp(img_tensor + noise, self.clip_min, self.clip_max)
            text_pos = self.ocr_reader.recognize(tensor_to_pil(adv_pos))
            score_pos = self._compute_score(text_pos, target_words_set, original_words_set)
            
            # 负向扰动
            adv_neg = torch.clamp(img_tensor - noise, self.clip_min, self.clip_max)
            text_neg = self.ocr_reader.recognize(tensor_to_pil(adv_neg))
            score_neg = self._compute_score(text_neg, target_words_set, original_words_set)
            
            # 梯度估计（最大化score）
            grad += (score_pos - score_neg) * noise / (2 * delta)
        
        return grad / num_samples
    
    def _compute_score(self, text: str, target_words: set, original_words: set) -> float:
        """计算识别文本与目标单词的匹配分数"""
        text_words = set(re.sub(r'[^a-zA-Z\s]', '', text).lower().split())
        target_matches = len(target_words.intersection(text_words))
        original_remaining = len(original_words.intersection(text_words))
        return target_matches - 0.3 * original_remaining


# ============================================================
# 结果分析与保存
# ============================================================
class ResultAnalyzer:
    """结果分析与保存"""
    
    def __init__(self, output_dir: str, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_word_success(
        self,
        original_text: str,
        target_text: str,
        adv_text: str,
        replacements: List[Dict],
    ) -> Tuple[int, int, List[Dict]]:
        """计算单词攻击成功率"""
        adv_words = re.sub(r'[^a-zA-Z\s]', '', adv_text).lower().split()
        adv_words_set = set(adv_words)
        
        success_count = 0
        results = []
        
        for repl in replacements:
            target_word = repl['target']
            target_clean = re.sub(r'[^a-zA-Z]', '', target_word).lower()
            
            success = target_clean in adv_words_set
            if success:
                success_count += 1
            
            results.append({
                **repl,
                'success': success,
            })
        
        return success_count, len(replacements), results
    
    def save_original_and_watermarked(
        self,
        idx: int,
        original: Image.Image,
        watermarked: Image.Image,
    ):
        """保存原始图像和水印图像（攻击前）"""
        original.save(os.path.join(self.output_dir, f'img_{idx:02d}_original.png'))
        watermarked.save(os.path.join(self.output_dir, f'img_{idx:02d}_watermarked.png'))
    
    def save_adversarial_and_residual(
        self,
        idx: int,
        watermarked: Image.Image,
        adversarial: Image.Image,
    ):
        """保存对抗图像和残差图（攻击后）"""
        adversarial.save(os.path.join(self.output_dir, f'img_{idx:02d}_adversarial.png'))
        
        wm_arr = np.array(watermarked.convert('RGB'), dtype=np.float32)
        adv_arr = np.array(adversarial.convert('RGB'), dtype=np.float32)
        
        residual = np.abs(adv_arr - wm_arr)
        residual_sum = residual.sum(axis=2)
        if residual_sum.max() > 0:
            residual_norm = (residual_sum / residual_sum.max() * 255).astype(np.uint8)
        else:
            residual_norm = np.zeros_like(residual_sum, dtype=np.uint8)
        
        residual_img = Image.fromarray(residual_norm, mode='L')
        residual_img.save(os.path.join(self.output_dir, f'img_{idx:02d}_residual.png'))
    
    def log_summary(self, results: List[Dict], no_attack: bool = False):
        """输出总结
        
        Args:
            results: 攻击结果列表
            no_attack: 是否为无攻击模式（nb_iter=0）
        """
        total_time = sum(r['time'] for r in results)
        
        self.logger.info("")
        self.logger.info("=" * 60)
        if no_attack:
            self.logger.info("SUMMARY (No Attack Mode)")
        else:
            self.logger.info("ATTACK SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total images: {len(results)}")
        
        if not no_attack:
            total_words = sum(r['total_words'] for r in results)
            success_words = sum(r['success_words'] for r in results)
            asr = success_words / total_words * 100 if total_words > 0 else 0
            self.logger.info(f"Total target words: {total_words}")
            self.logger.info(f"Successful attacks: {success_words}")
            self.logger.info(f"Word Attack Success Rate (ASR): {asr:.2f}%")
        
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 60)
        
        # 输出每张图的OCR结果汇总
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("OCR RESULTS FOR EACH IMAGE")
        self.logger.info("=" * 60)
        for r in results:
            idx = r['index']
            self.logger.info(f"\n--- Image {idx + 1} ---")
            self.logger.info(f"[Original OCR]:")
            self.logger.info(f"  {r.get('original_ocr', 'N/A')}")
            self.logger.info(f"[Watermarked OCR]:")
            self.logger.info(f"  {r.get('watermarked_ocr', 'N/A')}")
            
            # 只有在攻击模式下才输出对抗OCR和单词替换信息
            if not no_attack:
                self.logger.info(f"[Adversarial OCR]:")
                self.logger.info(f"  {r.get('adversarial_ocr', 'N/A')}")
                
                if r['replacements']:
                    self.logger.info(f"[Word Replacements]: {r['success_words']}/{r['total_words']} successful")
                    for repl in r['replacements']:
                        status = "OK" if repl['success'] else "FAIL"
                        self.logger.info(f"    {repl['original']} -> {repl['target']} [{status}]")


# ============================================================
# 提取图像中的单词
# ============================================================
def extract_text_lines(image: Image.Image, ocr_reader: EasyOCRWrapper) -> List[Dict]:
    """提取图像中的文本行信息"""
    results = ocr_reader.recognize_with_boxes(image)
    
    lines = []
    for item in results:
        if len(item) >= 3:
            bbox, text, conf = item[0], item[1], item[2]
        else:
            continue
        
        pts = np.array(bbox)
        y_min = int(pts[:, 1].min())
        y_max = int(pts[:, 1].max())
        x_min = int(pts[:, 0].min())
        x_max = int(pts[:, 0].max())
        
        lines.append({
            'text': text,
            'y_start': y_min,
            'y_end': y_max,
            'x_start': x_min,
            'x_end': x_max,
            'height': y_max - y_min,
            'confidence': conf,
        })
    
    lines.sort(key=lambda x: x['y_start'])
    
    return lines


# ============================================================
# 主攻击流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Paragraph Adversarial Watermark Attack on STR (PyTorch)')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to generate')
    parser.add_argument('--output_dir', type=str, default='results/eval_adversarial/FAWA/attacked', help='Output directory')
    parser.add_argument('--eps', type=float, default=0.2, help='Maximum perturbation (epsilon), 0.2 = 51/255')
    parser.add_argument('--eps_iter', type=float, default=5/255, help='Step size per iteration, 5/255 in [0,1] range')
    parser.add_argument('--nb_iter', type=int, default=0, help='Number of attack iterations')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--patch_size', type=int, default=30, help='Patch size for text generation')
    
    # 水印相关参数
    parser.add_argument('--wm_text', type=str, default='TDSC', help='Watermark text')
    parser.add_argument('--wm_color_r', type=int, default=250, help='Watermark color R')
    parser.add_argument('--wm_color_g', type=int, default=200, help='Watermark color G')
    parser.add_argument('--wm_color_b', type=int, default=150, help='Watermark color B')
    parser.add_argument('--wm_size', type=int, default=60, help='Watermark font size')
    parser.add_argument('--wm_angle', type=int, default=10, help='Watermark rotation angle')
    parser.add_argument('--wm_spacing_x', type=int, default=50, help='Watermark horizontal spacing')
    parser.add_argument('--wm_spacing_y', type=int, default=50, help='Watermark vertical spacing')
    
    # 图像尺寸参数
    parser.add_argument('--img_width_min', type=int, default=960, help='Minimum image width')
    parser.add_argument('--img_width_max', type=int, default=960, help='Maximum image width')
    parser.add_argument('--img_height_min', type=int, default=960, help='Minimum image height')
    parser.add_argument('--img_height_max', type=int, default=960, help='Maximum image height')
    
    # 覆盖阈值参数
    parser.add_argument('--coverage_threshold', type=float, default=0.3, help='Watermark coverage threshold for word selection (0.3 = 30%)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    
    logger, log_file = setup_logger(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("Paragraph Adversarial Watermark Attack on STR (PyTorch)")
    logger.info("Based on: Fast-Adversarial-Watermark-Attack-on-OCR")
    logger.info("Adapted for: UAW framework")
    logger.info("=" * 60)
    logger.info(f"Number of images: {args.num_images}")
    logger.info(f"Image size: width=[{args.img_width_min}, {args.img_width_max}], height=[{args.img_height_min}, {args.img_height_max}]")
    logger.info(f"Watermark: text='{args.wm_text}', size={args.wm_size}, angle={args.wm_angle}")
    logger.info(f"Watermark color: ({args.wm_color_r}, {args.wm_color_g}, {args.wm_color_b})")
    logger.info(f"Watermark spacing: x={args.wm_spacing_x}, y={args.wm_spacing_y}")
    logger.info(f"Device: {device}")
    logger.info(f"Epsilon: {args.eps}")
    logger.info(f"Epsilon iter: {args.eps_iter:.6f}")
    logger.info(f"Iterations: {args.nb_iter}")
    logger.info(f"Coverage threshold: {args.coverage_threshold:.0%}")
    logger.info(f"Log file: {log_file}")
    
    # 1. 创建数据集
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 1: Generating dataset using ImageDataset...")
    logger.info("=" * 60)
    
    to_tensor = transforms.ToTensor()
    dataset = ImageDataset(
        transform=to_tensor,
        length=args.num_images,
        adv_patch_size=(1, 3, args.patch_size, args.patch_size),
        test=True,
        width_range=(args.img_width_min, args.img_width_max),
        height_range=(args.img_height_min, args.img_height_max),
    )
    
    logger.info(f"Generated {len(dataset)} text images")
    
    # 2. 创建水印生成器
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: Creating watermark generator...")
    logger.info("=" * 60)
    
    wm_generator = WatermarkGenerator(
        text=args.wm_text,
        color=(args.wm_color_r, args.wm_color_g, args.wm_color_b),
        size=args.wm_size,
        angle=args.wm_angle,
    )
    
    logger.info(f"Watermark: '{args.wm_text}', size={args.wm_size}, color=({args.wm_color_r},{args.wm_color_g},{args.wm_color_b})")
    
    # 3. 创建OCR和攻击器
    logger.info("")
    logger.info("=" * 60)
    if args.nb_iter == 0:
        logger.info("Step 3: Initializing EasyOCR (no attacker needed)...")
    else:
        logger.info("Step 3: Initializing EasyOCR and attacker...")
    logger.info("=" * 60)
    
    ocr_reader = EasyOCRWrapper(device=args.device)
    
    # 只有在需要攻击时才创建攻击器和单词替换器
    if args.nb_iter > 0:
        attacker = AdversarialAttacker(
            ocr_reader=ocr_reader,
            device=device,
            eps=args.eps,
            eps_iter=args.eps_iter,
            nb_iter=args.nb_iter,
        )
        word_replacer = WordReplacer()
        logger.info("EasyOCR and attacker initialized for STR attack")
    else:
        attacker = None
        word_replacer = None
        logger.info("EasyOCR initialized (no attack mode)")
    
    # 4. 创建工具
    analyzer = ResultAnalyzer(args.output_dir, logger)
    
    # 5. 执行攻击
    logger.info("")
    logger.info("=" * 60)
    if args.nb_iter == 0:
        logger.info("Step 4: Saving images (no attack, nb_iter=0)...")
    else:
        logger.info("Step 4: Executing attacks...")
    logger.info("=" * 60)
    
    all_results = []
    
    for idx in range(len(dataset)):
        logger.info("")
        logger.info(f"--- Image {idx + 1}/{len(dataset)} ---")
        
        start_time = time.time()
        
        orig_image = dataset[idx]
        orig_image_pil = tensor_to_pil(orig_image)
        logger.info(f"Image size: {orig_image_pil.size}")
        
        # OCR识别原始图像
        original_text = ocr_reader.recognize(orig_image_pil)
        logger.info(f"[Original Image OCR]:")
        logger.info(f"  {original_text}")
        
        # 添加水印
        wm_image, wm_mask, positions = wm_generator.add_watermark_to_image(
            orig_image_pil, 
            spacing_x=args.wm_spacing_x, 
            spacing_y=args.wm_spacing_y
        )
        logger.info(f"Watermarks placed: {len(positions)}")
        
        # OCR识别水印图像
        watermarked_text = ocr_reader.recognize(wm_image)
        logger.info(f"[Watermarked Image OCR]:")
        logger.info(f"  {watermarked_text}")
        
        # 保存原图和水印图
        analyzer.save_original_and_watermarked(idx, orig_image_pil, wm_image)
        logger.info(f"Saved original and watermarked images")
        
        # 如果nb_iter为0，跳过攻击，只保存图像和OCR结果
        if args.nb_iter == 0:
            duration = time.time() - start_time
            all_results.append({
                'index': idx,
                'total_words': 0,
                'success_words': 0,
                'time': duration,
                'replacements': [],
                'original_ocr': original_text,
                'watermarked_ocr': watermarked_text,
                'adversarial_ocr': None,  # 无攻击时不记录对抗OCR结果
            })
            logger.info(f"Time: {duration:.2f}s (no attack)")
            continue
        
        text_lines = extract_text_lines(orig_image, ocr_reader)
        logger.info(f"Words detected: {len(text_lines)}")
        
        all_replacements = []
        
        for line_idx, line_info in enumerate(text_lines):
            covered = word_replacer.find_covered_words(
                line_info['text'],
                wm_mask,
                line_y_start=line_info['y_start'],
                line_y_end=line_info['y_end'],
                line_x_start=line_info['x_start'],
                line_x_end=line_info['x_end'],
                coverage_threshold=args.coverage_threshold,
            )
            
            if covered:
                _, replacements = word_replacer.replace_words_in_text(line_info['text'], covered)
                for r in replacements:
                    r['line_text'] = line_info['text']
                    r['line_idx'] = line_idx
                    all_replacements.append(r)
                    # 显示替换信息，包含原词->目标词
                    logger.info(f"  [{line_idx + 1}] '{r['original']}'->'{r['target']}' x=[{line_info['x_start']}, {line_info['x_end']}], y=[{line_info['y_start']}, {line_info['y_end']}], coverage: {r['coverage']:.2%}")
        
        if not all_replacements:
            logger.info("  No target words found, skipping attack")
            logger.info(f"[Adversarial Image OCR]: (same as watermarked)")
            logger.info(f"  {watermarked_text}")
            # 保存对抗图和残差图（与水印图相同）
            analyzer.save_adversarial_and_residual(idx, wm_image, wm_image)
            all_results.append({
                'index': idx,
                'total_words': 0,
                'success_words': 0,
                'time': time.time() - start_time,
                'replacements': [],
                'original_ocr': original_text,
                'watermarked_ocr': watermarked_text,
                'adversarial_ocr': watermarked_text,
            })
            continue
        
        logger.info(f"Target words to replace: {len(all_replacements)}")
        
        # 构建target_text（将所有目标单词组合）
        # 简单方式：用目标单词替换原始文本中的对应单词
        target_text = original_text
        for r in all_replacements:
            # 替换原始单词为目标单词
            target_text = re.sub(r'\b' + re.escape(r['original']) + r'\b', r['target'], target_text, count=1)
        
        logger.info(f"Original text: {original_text[:100]}...")
        logger.info(f"Target text: {target_text[:100]}...")
        
        adv_image, iterations, l2_norm, linf_norm, adv_text = attacker.attack_image(
            wm_image,
            wm_mask,
            all_replacements,
            original_text,
            target_text,
            logger,
        )
        
        success_count, total_count, repl_results = analyzer.compute_word_success(
            original_text, target_text, adv_text, all_replacements
        )
        
        duration = time.time() - start_time
        asr = success_count / total_count * 100 if total_count > 0 else 0
        
        # 输出对抗水印图的OCR识别结果
        logger.info(f"[Adversarial Image OCR]:")
        logger.info(f"  {adv_text}")
        
        logger.info(f"Attack results:")
        logger.info(f"  Iterations: {iterations}")
        logger.info(f"  L2 norm: {l2_norm:.4f}")
        logger.info(f"  Linf norm: {linf_norm:.4f}")
        logger.info(f"  Word ASR: {success_count}/{total_count} ({asr:.1f}%)")
        logger.info(f"  Time: {duration:.2f}s")
        
        logger.info(f"Word replacement details:")
        for r in repl_results:
            status = "OK" if r['success'] else "FAIL"
            logger.info(f"    {r['original']} -> {r['target']} [{status}]")
        
        # 攻击完成后保存对抗图和残差图
        analyzer.save_adversarial_and_residual(idx, wm_image, adv_image)
        logger.info(f"Saved adversarial and residual images")
        
        all_results.append({
            'index': idx,
            'total_words': total_count,
            'success_words': success_count,
            'time': duration,
            'replacements': repl_results,
            'l2_norm': l2_norm,
            'linf_norm': linf_norm,
            'original_ocr': original_text,
            'watermarked_ocr': watermarked_text,
            'adversarial_ocr': adv_text,
        })
    
    # 6. 输出总结
    analyzer.log_summary(all_results, no_attack=(args.nb_iter == 0))
    
    logger.info("")
    if args.nb_iter == 0:
        logger.info("Image generation completed (no attack)!")
    else:
        logger.info("Attack completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Log file: {log_file}")


if __name__ == '__main__':
    main()
