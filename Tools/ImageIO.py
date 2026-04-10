import torch
import cv2
from torchvision import transforms
from AllConfig.GConfig import test_img_path
from PIL import Image
import numpy as np
import logging
import os

def logger_config(log_filename, logging_name='mylog'):
    log_path=os.path.dirname(log_filename)
    os.makedirs(log_path, mode=0o755, exist_ok=True)
    # 获取logger对象
    logger = logging.getLogger(logging_name)
    logger.setLevel(logging.INFO)
    # 创建一个handler,用于写入日志文件
    fh = logging.FileHandler(log_filename, mode='a+')
    fh.setFormatter(logging.Formatter("[%(asctime)s]:%(levelname)s:%(message)s"))
    logger.addHandler(fh)
    # 创建一个handler，输出到控制台
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger

def logInfo(log, file_path, console_print=True, file_print=True):
    f = open(file_path, 'w')
    if console_print:
        print(log)
    if file_print:
        print(log, file=f, flush=True)

def logINFO(log, f):
    print(log)
    print(log, file=f, flush=True)

def img_read(image_path, device=None) -> torch.Tensor:
    transform = transforms.ToTensor()
    if os.path.exists(image_path):
        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
    elif os.path.exists(image_path.replace('_ti.png', '.png')):
        im = cv2.imread(image_path.replace('_ti.png', '.png'), cv2.IMREAD_COLOR)
    else:
        raise Exception('{} does not exist'.format(image_path))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # img_show3(im)
    img = transform(im)
    img = img.unsqueeze_(0)
    return img

def img_write(image_path: str, image_tensor: torch.Tensor):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    arr = image_tensor.permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    im.save(image_path)

def extract_background_mask(image_tensor: torch.Tensor) -> torch.Tensor:
    """提取背景掩码（文字区域为0，背景为1）"""
    if image_tensor.dim() == 4:
        gray = image_tensor.mean(dim=1, keepdim=True)
    else:
        gray = image_tensor.mean(dim=0, keepdim=True).unsqueeze(0)
    
    mask = (gray > 0.85).float()
    return mask

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """将Tensor转换为PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """将PIL Image转换为Tensor"""
    arr = np.array(image.convert('RGB'))
    tensor = torch.from_numpy(arr).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor

def tensor_to_cv2(img_tensor: torch.Tensor):
    assert (len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1)
    img_tensor = img_tensor.clone().detach().cpu()
    img_tensor = img_tensor.squeeze()
    img_tensor = img_tensor.mul_(255).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    img_cv = cv2.cvtColor(img_tensor, cv2.COLOR_RGB2BGR)
    return img_cv

def cv2_to_tensor(cv2_img, device='cpu'):
    """
    将OpenCV图像(BGR, uint8)转换为PyTorch张量(RGB, float32, [C,H,W])
    """
    # [H,W,C] -> [C,H,W], 归一化到[0,1]
    tensor = torch.from_numpy(cv2_img).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)
