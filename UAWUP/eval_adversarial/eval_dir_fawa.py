# eval_dir_fawa.py
import numpy as np
# ---- NumPy 1.24+ compat hotfix (must be before importing PAN) ----
if not hasattr(np, "bool"):
    np.bool = np.bool_
import sys
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/UAW/")
import torch
from model_CRAFT.pred_single import *
from Tools.ImageIO import *
from Tools.Baseimagetool import *
import os
import easyocr
from mmocr.utils.ocr import MMOCR
from model_PAN.pred_single import pan_pred_single
from Tools.PANTools import *
from Tools.CRAFTTools import *
from Tools.DBTools import *
import cv2
from UDUP.Auxiliary import *
from Tools.EvalTool import *
from tqdm import tqdm
from UAWUP.dataset import ImageDataset
from torch.utils.data import DataLoader
import warnings
import re
warnings.filterwarnings("ignore")

evaluator=DetectionIoUEvaluator()

def get_result_easyocr(result):
    points=[]
    for item in result:
        points.append(item[0])
    return np.array(points)

def logINFO(log, f, concole_print=True, file_print=True):
    if concole_print:
        print(log)
    if file_print:
        print(log, file=f, flush=True)

def get_img_list(dir, exp=r'^\d+_awti\.png$'):
    images_name = []
    for filename in os.listdir(dir):
        if re.match(exp, filename):
            images_name.append(filename)
    return images_name

def evaluate_and_draw(model, img_dir, log_file, save_dir, pan_cfg=None, model_name="CRAFT"):
    global evaluator
    image_names = get_img_list(img_dir, exp=r'^img_\d+_watermarked\.png$')
    merge_images = [img_read(os.path.join(img_dir, name))[:, :, :960, :960] for name in image_names]
    images_gt = [img_read(os.path.join(img_dir, name.replace('_watermarked', '_original')))[:, :, :960, :960] for name in image_names]

    results = []  # PRF
    for img_name, merge_image, img in tqdm(zip(image_names, merge_images, images_gt)):
        temp_save_path = os.path.join(save_dir, img_name)
        
        merge_image = merge_image.to('cuda:0')
        img = img.to('cuda:0')
        
        if model_name == 'DBnet' or model_name == 'PanNet' or model_name == 'PSENet':
            tmp_merge_image = tensor_to_cv2(merge_image)
            tmp_img = tensor_to_cv2(img)
            boxes = get_pred_boxes_formmocr(model.readtext(tmp_merge_image))
            gt_boxes = get_pred_boxes_formmocr(model.readtext(tmp_img))
        elif model_name == 'CRAFT':
            (score_text, score_link, target_ratio) = single_grad_inference(model, merge_image, [], model_name, is_eval=True)
            boxes = get_CRAFT_box(score_text, score_link, target_ratio, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
            (gt_score_text, gt_score_link, gt_target_ratio) = single_grad_inference(model, img, [], model_name, is_eval=True)
            gt_boxes = get_CRAFT_box(gt_score_text, gt_score_link, gt_target_ratio, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
        elif model_name == 'EasyOCR':
            tmp_merge_image = tensor_to_cv2(merge_image)
            res = model.readtext(tmp_merge_image)
            boxes = get_result_easyocr(res)
            tmp_img = tensor_to_cv2(img)
            gt_res = model.readtext(tmp_img)
            gt_boxes = get_result_easyocr(gt_res)
        elif model_name == 'PanPP':
            data = pan_preprocess_image(merge_image)
            boxes = pan_pred_single(model, pan_cfg, data)
            boxes = get_pred_boxes_forpanpp(boxes)
            gt_data = pan_preprocess_image(img)
            gt_boxes = pan_pred_single(model, pan_cfg, gt_data)
            gt_boxes = get_pred_boxes_forpanpp(gt_boxes)
        # gt = read_txt(gt)  # Compatible with old interfaces
        # results.append(evaluator.evaluate_image(gt, boxes))
        results.append(evaluator.evaluate_image(gt_boxes, boxes))
        # draw box
        Draw_box(tensor_to_cv2(img), gt_boxes, temp_save_path.replace('_watermarked.png', '_gt.png'), model_name=model_name)
        Draw_box(tensor_to_cv2(merge_image), boxes, temp_save_path, model_name=model_name)
        P, R, F = evaluator.combine_results(results)
        logINFO("img_name:{} P:{:.4f} R:{:.4f} F:{:.4f}".format(img_name, P, R, F), log_file, concole_print=False)
    P, R, F = evaluator.combine_results(results)
    return P, R, F

def main(img_dir, log_file, save_dir, model_name):
    pan_cfg = None
    config = '/home/dongli911/anaconda3/envs/udup/lib/python3.8/site-packages/mmocr/configs/textdet/'
    if model_name == 'CRAFT':
        model = load_CRAFTmodel()
    elif model_name == 'DBnet':
        config = os.path.join(config, 'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py')
        model = MMOCR(det='DB_r18', det_config=config, recog=None)
    elif model_name == 'EasyOCR':
        model = easyocr.Reader(['en'], gpu=True, model_storage_directory='AllConfig/all_model')
    elif model_name == 'PanPP':
        model, pan_cfg = load_PANPlusmodel()
    elif model_name == 'PanNet':
        config = os.path.join(config, 'panet/panet_r18_fpem_ffm_600e_icdar2015.py')
        model = MMOCR(det='PANet_IC15', det_config=config, recog=None)
    elif model_name == 'PSENet':
        config = os.path.join(config, 'psenet/psenet_r50_fpnf_600e_icdar2015.py')
        model = MMOCR(det='PS_IC15', det_config=config, recog=None)
    
    P, R, F = evaluate_and_draw(model, img_dir, log_file, save_dir, pan_cfg, model_name)
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
    model_name = ['CRAFT', 'DBnet', 'EasyOCR', 'PanPP', 'PSENet']
    img_dir = f"results/eval_adversarial/FAWA/wti"
    save_root = f"results/eval_adversarial/FAWA"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    for model in model_name:
        save_dir = os.path.join(save_root, model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                os.remove(file_path)

        log_path = os.path.join(save_dir, 'log')
        log_file = open(log_path, 'a')
        P, R, F = main(img_dir, log_file, save_dir, model)
        logINFO("{}:{:.4f}/{:.4f}/{:.4f}".format(model, R, P, F), log_file)
        log_file.close()
