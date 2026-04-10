import os
import cv2
import torch
from torch.utils.data import Dataset
import random
import textwrap
import numpy as np
import nltk
from PIL import Image, ImageDraw, ImageFont


def generate_random_text():
    """
    使用NLTK Brown语料库生成随机英文文本段落。
    
    返回:
        str: 随机文本字符串
    """
    words = nltk.corpus.brown.words()
    strlen = random.randint(1000, 10000)
    start = random.randint(0, len(words) - strlen)
    # Initialize an empty list to hold the words in the paragraph
    paragraph_words = words[start: start + strlen]
    paragraph = []
    paragraph.append(paragraph_words[0])
    for word in paragraph_words[1:]:
        if not word.isalnum():
            paragraph[-1] += word
        else:
            paragraph.append(word)
    # Convert the list of words into a single string
    paragraph = " ".join(paragraph)
    return paragraph


def generate_random_textimg(s, width=960, height=960, font_size=None, font_color=None, opt=False, document=False, device=None):
    input_text = generate_random_text()
    if isinstance(width, tuple):
        width = random.randint(*width)
    if isinstance(height, tuple):
        height = random.randint(*height)
    text_image = Image.new("RGB", (width, height), color="white")

    # Set the font properties
    font_folder = "/usr/share/fonts/truetype/tlwg"
    font_files = [f for f in os.listdir(font_folder) if f.endswith(".ttf") or f.endswith(".otf")]
    if not font_files:
        raise ValueError("No font files found in the specified folder!")
    font_name = os.path.join(font_folder, random.choice(font_files))

    # 定义行高和行间距
    if document == False:
        r = random.randint(0, s//3)
        font_size = int(s + r)  # 行高设为s
        line_height = font_size  # 匹配s
        line_spacing = int(s - r)
    else:
        font_size = random.randint(15, 45)  # 常见像素高度px为15-45，Word的字体大小单位是磅pt，1pt≈1.33px
        line_height = font_size  # 匹配s
        line_spacing = int(font_size*0.8)

    # 设置字体颜色
    eps = 100
    if font_color is None:
        if opt:
            if random.random() < 0:  # 都是黑色字
                R = random.randrange(0, 255-eps)
                G = random.randrange(0, 255-eps)
                B = random.randrange(0, 255-eps)
                font_color = (R, G, B)
            else:
                font_color = (0, 0, 0)
        else:
            if random.random() < 0.2:  # 现实黑色字占比大
                R = random.randrange(0, 255-eps)
                G = random.randrange(0, 255-eps)
                B = random.randrange(0, 255-eps)
                font_color = (R, G, B)
            else:
                font_color = (0, 0, 0)
    
    font = ImageFont.truetype(font_name, font_size)

    # Create a drawing context
    draw = ImageDraw.Draw(text_image)

    # 逐行绘制文本
    x, y = s, s  # 起始坐标，带r像素边距
    words = input_text.split()
    current_line = []
    line_width = 0
    lines = []
    
    # 手动换行以适应图像宽度
    for word in words:
        word_width = font.getlength(word + " ")
        if line_width + word_width <= width - s//2:  # s//2像素边距
            current_line.append(word)
            line_width += word_width
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            line_width = word_width

    if current_line:
        lines.append(" ".join(current_line))

    # 根据文字高度自适应起始坐标
    bbox = draw.textbbox((x, y), lines[0], font=font)
    # print(bbox)  # 左，上，右，下
    y -= bbox[1] - s  # 不知道为什么会向下偏移，这里做个补偿
    
    # 绘制每行文本
    for line in lines:
        if y + line_height <= height:  # 确保文本不超过图像高度
            draw.text((x, y), line, font=font, fill=font_color)
            y += line_height + line_spacing
    
    # 转换为torch.Tensor BGR格式
    text_image_cv = cv2.cvtColor(np.array(text_image), cv2.COLOR_RGB2BGR)
    return text_image_cv


class ImageDataset(Dataset):
    def __init__(self, transform=None, length=100, adv_patch_size=None, test=False, width_range=(960, 960), height_range=(960, 960)):
        self.transform = transform
        self.length = length
        self.test = test
        self.adv_patch_size = adv_patch_size
        self.width_range = width_range
        self.height_range = height_range
    
    def __getitem__(self, index):
        if self.test:
            random.seed(index)
            torch.manual_seed(index)
            np.random.seed(index)
        else:
            random.seed()  # 让 Python random 回到系统默认
            torch.manual_seed(torch.initial_seed())  # 让 torch 继续随机
            np.random.seed()  # 让 numpy 继续随机

        textimg = generate_random_textimg(self.adv_patch_size[-1], width=self.width_range, height=self.height_range, opt=True)
        if self.transform:
            textimg = self.transform(textimg)
        # bits = torch.tensor([random.randint(0, 1) for _ in range(64)], dtype=torch.float32)
        return textimg

    def __len__(self):
        return self.length
