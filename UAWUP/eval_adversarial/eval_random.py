# 黑盒 STD 的迁移性
# 测试一对底纹的对抗性
import numpy as np
# ---- NumPy 1.24+ compat hotfix (must be before importing PAN) ----
if not hasattr(np, "bool"):
    np.bool = np.bool_
import sys
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/PycharmProjects/UAW")
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
from watermarklab.noiselayers.testdistortions import *
from tqdm import tqdm
from UAWUP.dataset import ImageDataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import torchvision.transforms as transforms

evaluator=DetectionIoUEvaluator()

def logINFO(log, f):
    print(log, file=f, flush=True)
    print(log)

def get_result_easyocr(result):
    points=[]
    for item in result:
        points.append(item[0])
    return np.array(points)

def get_result_mmocr(result):
    points=[]
    for item in result:
        if isinstance(item,(np.ndarray)):
            points.append(item.tolist())
        else:
            points.append(item)
    return points



def evaluate_and_draw(model, adv_patch1, adv_patch2, save_path, pan_cfg=None, model_name="CRAFT"):
    global evaluator
    to_tensor = transforms.ToTensor()
    test_dataset = ImageDataset(transform=to_tensor, length=100, adv_patch_size=adv_patch1.shape, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    # image_names = os.listdir(image_root)
    # selected_indices = random.sample(range(len(image_names)), 70)
    # image_names = [image_names[i] for i in selected_indices]

    results = []  # PRF
    # for name in tqdm(image_names):
    for i, img in tqdm(enumerate(test_dataloader)):
        # img = img_read(os.path.join(image_root, name))
        temp_save_path = os.path.join(save_path, '{:03d}.png'.format(i+1))
        h, w = img.shape[2:]
        if adv_patch1 != None:
            UAU = repeat_2patch_flip(adv_patch1.clone().detach(), adv_patch2.clone().detach(), h, w)
            mask_t = extract_background(img).to('cuda:0')
            img = img.to('cuda:0')
            merge_image = img * (1 - mask_t) + mask_t * UAU
        else:
            merge_image = img.clone().to('cuda:0')
        # cv2.imwrite(temp_save_path.replace('jpeg_resize', 'jpeg_resize_DU'), tensor_to_cv2(merge_image))
        merge_image = merge_image.to('cuda:0')
        
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
             
        results.append(evaluator.evaluate_image(gt_boxes, boxes))
        # draw
        Draw_box(tensor_to_cv2(img), gt_boxes, temp_save_path.replace('.png', '_gt.png'), model_name=model_name)
        Draw_box(tensor_to_cv2(merge_image), boxes, temp_save_path, model_name=model_name)
    P, R, F = evaluator.combine_results(results)
    return P, R, F

def main(patch_path, save_path, model_name):
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

    if patch_path != None:
        if patch_path.endswith('.png'):
            # 定义 transform，把 PIL 图片转成 Tensor，并保持 [0,1] 范围
            to_tensor = transforms.ToTensor()
            # 读取 advpatch1.png
            adv_patch1 = Image.open(patch_path).convert("RGB")
            adv_patch1 = to_tensor(adv_patch1).unsqueeze(0).cuda()  # (1, C, H, W)
            # 读取 advpatch2.png
            adv_patch2_path = patch_path.replace('patch1', 'patch2')
            adv_patch2 = Image.open(adv_patch2_path).convert("RGB")
            adv_patch2 = to_tensor(adv_patch2).unsqueeze(0).cuda()  # (1, C, H, W)

        else:
            adv_patch1 = torch.load(patch_path).cuda()
            adv_patch2 = torch.load(patch_path.replace('advpatch1', 'advpatch2')).cuda()
    else:
        adv_patch1 = None
        adv_patch2 = None
    P, R, F = evaluate_and_draw(model, adv_patch1, adv_patch2, save_path, pan_cfg ,model_name=model_name)
    # print("R:{},P:{},F:{}".format(R,P,F))
    return P, R, F

def gen_iter_dict(root_eval, special_eval_item):
    mypath=os.path.join(root_eval ,special_eval_item)
    dir_list=os.listdir(mypath)
    is_need=[]
    for item in dir_list:
        if item.isdigit():
            is_need.append(item)
    return is_need

if __name__=="__main__":
    model_list = ['CRAFT', 'PanPP']  # ['CRAFT', 'DBnet', 'EasyOCR', 'PanPP', 'PanNet', 'PSENet']
    patch_path = 'eval_watermark/size_it/awu/random_patch1.png'
    for model_name in model_list:
        save_path_root = os.path.join("eval_adversarial/results_transfer", model_name)
        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)
        log_path = os.path.join(save_path_root, 'log')
        log_file = open(log_path, 'a')
        save_path = os.path.join(save_path_root, 'random')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            for filename in os.listdir(save_path):
                file_path = os.path.join(save_path, filename)
                os.remove(file_path)
        P, R, F = main(patch_path, save_path, model_name)
        logINFO("patch_size:30 random P:{:.4f} R:{:.4f} F:{:.4f}".format(P, R, F), log_file)
        log_file.close()
