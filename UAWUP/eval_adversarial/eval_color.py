# size_eps 测试
# 测试一对底纹的对抗性
import sys
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/UAW")
import torch
from model_CRAFT.pred_single import *
from Tools.ImageIO import *
from Tools.Baseimagetool import *
import os
from Tools.CRAFTTools import *
from Tools.DBTools import *
import cv2
from UDUP.Auxiliary import *
from Tools.EvalTool import *
from tqdm import tqdm
from dataset import ImageDataset
from torch.utils.data import DataLoader
import warnings
import re
warnings.filterwarnings("ignore")

evaluator=DetectionIoUEvaluator()


import os
import re
from collections import defaultdict

def group_images_by_color(folder_path):
    """
    将图像按颜色名称分类，返回字典格式结果
    输入：文件夹路径
    输出：{
        "color1": [有序路径列表],
        "color2": [有序路径列表],
        ...
    }
    """
    # 初始化颜色字典（自动处理不存在的key）
    color_dict = defaultdict(list)
    
    # 匹配颜色和编号的正则表达式（支持大小写字母、数字编号和常见图片格式）
    pattern = re.compile(
        r'^([a-zA-Z]+)_'   # 颜色名称（字母开头）
        r'(\d+)'          # 数字编号
        r'\.(?:png|jpg|jpeg)$',  # 图片格式
        re.IGNORECASE
    )
    
    # 遍历文件夹所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 跳过子目录和非文件项
        if not os.path.isfile(file_path):
            continue
            
        # 匹配文件名格式
        match = pattern.match(filename)
        if match:
            # 统一颜色名为小写，避免大小写差异
            color_name = match.group(1).lower()  
            # 提取数字编号用于排序
            number = int(match.group(2))         
            
            # 存储编号和路径的元组，便于后续排序
            color_dict[color_name].append((number, file_path))
    
    # 对每个颜色列表进行排序
    sorted_dict = {}
    for color, files in color_dict.items():
        # 按数字编号排序
        sorted_files = sorted(files, key=lambda x: x[0])
        # 仅保留路径信息
        sorted_paths = [path for (num, path) in sorted_files]
        sorted_dict[color] = sorted_paths
    print(sorted_dict)
    return sorted_dict

def evaluate_and_draw(model, adv_patch1, adv_patch2, img_path, log_file, save_path, model_name="CRAFT"):
    global evaluator
    
    # image_names, img_list, img_gt_list = get_img_list(img_path)
    imgs = [img_read(name) for name in img_path]
    image_names = [name.split('/')[-1] for name in img_path]

    results = []  # PRF
    for img_name, img in zip(image_names, imgs):
        # temp_save_path = os.path.join(save_path, name.split('/')[-1])
        temp_save_path = os.path.join(save_path, img_name)
        h, w = img.shape[2:]
        if adv_patch1 != None:
            UAU = repeat_2patch_flip(adv_patch1.clone().detach(), adv_patch2.clone().detach(), h, w)
            mask_t = extract_background(img).to('cuda:0')
            img = img.to('cuda:0')
            merge_image = img * (1 - mask_t) + mask_t * UAU
        else:
            merge_image = img.clone().to('cuda:0')
        merge_image = merge_image.to('cuda:0')
        if model_name == 'DBnet':
            preds = single_grad_inference(model, merge_image, [], model_name)
            preds = preds[0]
            dilates, boxes = get_DB_dilateds_boxes(preds, h, w, min_area=100)
        elif model_name == 'CRAFT':
            (score_text, score_link, target_ratio) = single_grad_inference(model, merge_image, [], model_name, is_eval=True)
            boxes = get_CRAFT_box(score_text, score_link, target_ratio, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
            (gt_score_text, gt_score_link, gt_target_ratio) = single_grad_inference(model, img, [], model_name, is_eval=True)
            gt_boxes = get_CRAFT_box(gt_score_text, gt_score_link, gt_target_ratio, text_threshold=0.7, link_threshold=0.4, low_text=0.4)

        results.append(evaluator.evaluate_image(gt_boxes, boxes))
        # draw
        Draw_box(tensor_to_cv2(img), gt_boxes, temp_save_path.replace('.png', '_gt.png'), model_name=model_name)
        Draw_box(tensor_to_cv2(merge_image), boxes, temp_save_path, model_name=model_name)
        P, R, F = evaluator.combine_results(results)
        # print("img_name:{} P:{:.4f} R:{:.4f} F:{:.4f}".format(img_name, P, R, F))
        logINFO("img_name:{} P:{:.4f} R:{:.4f} F:{:.4f}".format(img_name, P, R, F), log_file)
    P, R, F = evaluator.combine_results(results)
    return P, R, F

def main(patch_path, img_path, log_file, save_path, model_name):
    if model_name=='CRAFT':
        model = load_CRAFTmodel()
    else:
        model = load_DBmodel()
    if patch_path != None:
        adv_patch1 = torch.load(patch_path).cuda()
        adv_patch2 = torch.load(patch_path.replace('advpatch1', 'advpatch2')).cuda()
    else:
        adv_patch1 = None
        adv_patch2 = None
    P, R, F = evaluate_and_draw(model, adv_patch1, adv_patch2, img_path, log_file, save_path, model_name=model_name)
    # print("P:{},R:{},F:{}".format(R,P,F))
    return P, R, F

def gen_iter_dict(root_eval,special_eval_item):
    mypath=os.path.join(root_eval,special_eval_item)
    dir_list=os.listdir(mypath)
    is_need=[]
    for item in dir_list:
        if item.isdigit():
            is_need.append(item)
    return is_need

if __name__=="__main__":
    sele_size = 30
    lambdaw = 0.01
    iter = 27
    model_name = "CRAFT"
    patch_path = "results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01/advtorch/advpatch1_27"
    img_path = "AllData/test_color"     # 不同颜色字体的图
    save_path_root = "eval_adversarial/results_color/"
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    
    result = group_images_by_color(img_path)
    for color, img_path in result.items():
        print(f"\n{color.upper()} 颜色图像（共 {len(img_path)} 张）：")
        # print(paths)
        save_path = os.path.join(save_path_root, color)
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            for filename in os.listdir(save_path):
                file_path = os.path.join(save_path, filename)
                os.remove(file_path)
        log_path = os.path.join(save_path, 'log')
        log_file = open(log_path, 'a')
        P, R, F = main(patch_path, img_path, log_file, save_path, model_name)
        logINFO("patch_size:{} lambdaw:{} iter:{} P:{:.4f} R:{:.4f} F:{:.4f}".format(sele_size, lambdaw, iter, P, R, F), log_file)
        log_file.close()