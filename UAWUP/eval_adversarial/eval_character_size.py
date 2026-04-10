# 字符大小不可知
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

def logINFO(log, f):
    print(log, file=f, flush=True)
    print(log)


import os
import re
from collections import defaultdict

def get_images_path(folder_path):
    
    # 遍历文件夹所有文件
    img = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img.append(file_path)
    
    return img

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
    lambdaw_list = [1] # [10, 1, 0.01, 0.001, 0.01]
    patch_size = [20] # [10, 20, 30, 40, 50]
    iter_list = [25] # [25, 25, 27, 27, 28]
    model_name = "CRAFT"
    # patch_path = "results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01/advtorch/advpatch1_27"
    img_path_root = "AllData/character_size"     # 不同字符大小的图：Large、Normal、Tiny
    save_path_root = "eval_adversarial/results_character_size/"


    root_eval = os.path.join('results', 'AllData_results', 'results_uawup_eps100')
    special_eval=[
        "size=10_step=3_eps=100_lambdaw=10/advtorch/advpatch1_25",
        "size=20_step=3_eps=100_lambdaw=1/advtorch/advpatch1_25",
        "size=30_step=3_eps=100_lambdaw=0.01/advtorch/advpatch1_27",
        "size=40_step=3_eps=100_lambdaw=0.001/advtorch/advpatch1_27",
        "size=50_step=3_eps=100_lambdaw=0.01/advtorch/advpatch1_28"
    ]
    
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    log_path = os.path.join(save_path_root, 'log')
    log_file = open(log_path, 'a')
    img_paths = get_images_path(img_path_root)
    
    for path, s, iter, lambdaw in zip(special_eval, patch_size, iter_list, lambdaw_list):
        patch_path = os.path.join(root_eval, path)
        save_path = os.path.join(save_path_root, '{}_{}_{}'.format(s, lambdaw, iter))
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            for filename in os.listdir(save_path):
                file_path = os.path.join(save_path, filename)
                os.remove(file_path)
        P, R, F = main(patch_path, img_paths, log_file, save_path, model_name)
        logINFO("patch_size:{} lambdaw:{} iter:{} P:{:.4f} R:{:.4f} F:{:.4f}".format(s, lambdaw, iter, P, R, F), log_file)
    