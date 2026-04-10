# size_eps 测试
# 测试一对底纹的对抗性
import numpy as np
import sys
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/UAW/")
import torch
from model_CRAFT.pred_single import *
from Tools.ImageIO import *
from Tools.Baseimagetool import *
import os
from natsort import natsorted
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
import time
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

def evaluate_and_draw(model, img_dir, gt_img_dir, log_file, save_dir, pan_cfg=None, model_name="CRAFT", timing_dict=None, res=None, device=None):
    global evaluator
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []  # PRF
    image_times = []  # Track time per image

    image_names = natsorted([f for f in os.listdir(img_dir) if f.endswith('_awti.png')])
    for img_name in tqdm(image_names):
        img_start_time = time.perf_counter()  # Start timing for this image

        merge_image = img_read(os.path.join(img_dir, img_name))[:, :, :res, :res].to(device)
        img = img_read(os.path.join(gt_img_dir, img_name.replace('_awti', '_ti')))[ :, :, :res, :res].to(device)
        temp_save_path = os.path.join(save_dir, img_name)
        h, w = img.shape[2:]
        
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
        # gt = read_txt(gt)
        # results.append(evaluator.evaluate_image(gt, boxes))
        result = evaluator.evaluate_image(gt_boxes, boxes)
        results.append(result)
        P, R, F = evaluator.combine_results([result])
        logINFO("img_name:{} P:{:.4f} R:{:.4f} F:{:.4f}".format(img_name, P, R, F), log_file, concole_print=False)
        # 绘制结果
        Draw_box(tensor_to_cv2(img), gt_boxes, temp_save_path.replace('_awti.png', '_gt.png'), model_name=model_name)
        Draw_box(tensor_to_cv2(merge_image), boxes, temp_save_path, model_name=model_name)
        
        # Record time for this image
        img_end_time = time.perf_counter()
        image_times.append(img_end_time - img_start_time)
    
    # Calculate and store average time per image for this model
    if timing_dict is not None and image_times:
        timing_dict[model_name] = np.mean(image_times)
    
    P, R, F = evaluator.combine_results(results)
    return P, R, F

def main(img_dir, gt_img_dir, log_file, save_dir, model_name, timing_dict=None, res=None, device=None):
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
    
    P, R, F = evaluate_and_draw(model, img_dir, gt_img_dir, log_file, save_dir, pan_cfg, model_name, timing_dict, res, device)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA for computation')
    parser.add_argument('--res', type=int, default=None, help='Image resolution')
    parser.add_argument('--img_dir', type=str, default='results/eval_watermark/real/web_page', help='待检测图片目录')
    parser.add_argument('--gt_img_dir', type=str, default='AllData/web_page', help='文本图像目录')
    parser.add_argument('--save_root', type=str, default='results/eval_adversarial/real/web_page', help='检测结果保存目录')
    args = parser.parse_args()
    
    use_cuda = args.use_cuda
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_name = ['CRAFT']  # , 'DBnet', 'EasyOCR', 'PanPP', 'PSENet'
    img_dir = args.img_dir
    gt_img_dir = args.gt_img_dir
    save_root = args.save_root
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    model_timing = {}
    log_path = os.path.join(save_root, 'log')
    log_file = open(log_path, 'w')
    for model in model_name:
        save_dir = os.path.join(save_root, model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                os.remove(file_path)

        P, R, F = main(img_dir, gt_img_dir, log_file, save_dir, model, model_timing, args.res, device)
        logINFO("[res]{}[model]{}--[R]{:.4f}[P]{:.4f}[F]{:.4f}".format(args.res, model, R, P, F), log_file)
    
    # Print timing statistics for STD models
    if model_timing:
        logINFO("\n" + "="*60, log_file)
        logINFO("[Timing Statistics] STD Model Evaluation", log_file)
        logINFO("="*60, log_file)
        for model_name in sorted(model_timing.keys()):
            logINFO(f"{model_name}: {model_timing[model_name]:.4f} seconds per image", log_file)
        logINFO(f"Average across all models: {np.mean(list(model_timing.values())):.4f} seconds per image", log_file)
        logINFO("="*60 + "\n", log_file)
    log_file.close()
    