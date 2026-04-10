# 黑盒 STD 的迁移性
# 测试一对底纹的对抗性
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/UAW")

import torch
from torchvision import transforms
from Tools.ImageIO import *
from Tools.Baseimagetool import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import easyocr

from Tools.TextBoxesPPTools import *
from model_PAN.pred_single import pan_pred_single
from Tools.PANTools import *
from Tools.CRAFTTools import *
from Tools.DBTools import *
from Tools.EASTTools import *
from Tools.TCMTools import *
from UDUP.Auxiliary import *
from Tools.EvalTool import *
from watermarklab.noiselayers.testdistortions import *
from UAWUP.dataset import ImageDataset

evaluator = DetectionIoUEvaluator()


def get_result_easyocr(result):
    points = []
    for item in result:
        points.append(item[0])
    return np.array(points)


def get_result_mmocr(result):
    points = []
    for item in result:
        if isinstance(item, (np.ndarray)):
            points.append(item.tolist())
        else:
            points.append(item)
    return points


def evaluate_and_draw(model, adv_patch1, adv_patch2, save_path, pan_cfg=None, model_name="CRAFT"):
    global evaluator
    to_tensor = transforms.ToTensor()
    test_dataset = ImageDataset(transform=to_tensor, length=100, adv_patch_size=(30, 30), test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    results = []  # PRF
    for i, img in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        temp_save_path = os.path.join(save_path, "{:03d}.png".format(i + 1))
        h, w = img.shape[2:]
        img = img.to("cuda:0")
    
        if adv_patch1 is not None:
            UAU = repeat_2patch_flip(adv_patch1.clone().detach(), adv_patch2.clone().detach(), h, w)
            mask_t = extract_background(img).to("cuda:0")
            merge_image = img * (1 - mask_t) + mask_t * UAU
        else:
            merge_image = img.detach().clone()
    
        merge_image = merge_image.to("cuda:0")
    
        if model_name in ["DBnet", "PanNET", "PanNet", "PSENet", "TCM"]:
            tmp_merge_image = tensor_to_cv2(merge_image)
            tmp_img = tensor_to_cv2(img)
            boxes = get_pred_boxes_formmocr(model.readtext(tmp_merge_image))
            gt_boxes = get_pred_boxes_formmocr(model.readtext(tmp_img))
        elif model_name == "CRAFT":
            score_text, score_link, target_ratio = single_grad_inference(model, merge_image, [], model_name, is_eval=True)
            boxes = get_CRAFT_box(score_text, score_link, target_ratio, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
            gt_score_text, gt_score_link, gt_target_ratio = single_grad_inference(model, img, [], model_name, is_eval=True)
            gt_boxes = get_CRAFT_box(gt_score_text, gt_score_link, gt_target_ratio, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
        elif model_name == "EasyOCR":
            tmp_merge_image = tensor_to_cv2(merge_image)
            res = model.readtext(tmp_merge_image)
            boxes = get_result_easyocr(res)
            tmp_img = tensor_to_cv2(img)
            gt_res = model.readtext(tmp_img)
            gt_boxes = get_result_easyocr(gt_res)
        elif model_name == "PanPP":
            data = pan_preprocess_image(merge_image)
            boxes = pan_pred_single(model, pan_cfg, data)
            boxes = get_pred_boxes_forpanpp(boxes)
            gt_data = pan_preprocess_image(img)
            gt_boxes = pan_pred_single(model, pan_cfg, gt_data)
            gt_boxes = get_pred_boxes_forpanpp(gt_boxes)
        elif model_name == "TextBoxesPP":
            net, encoder, tb_cls_thresh, tb_nms_thresh = model
            boxes, scores = textboxespp_pred_tensor(net, encoder, merge_image, input_size=600, cls_thresh=tb_cls_thresh, nms_thresh=tb_nms_thresh)
            gt_boxes, gt_scores = textboxespp_pred_tensor(net, encoder, img, input_size=600, cls_thresh=tb_cls_thresh, nms_thresh=tb_nms_thresh)
        elif model_name == "EAST":
            boxes = east_tensor_to_boxes(model, merge_image)
            gt_boxes = east_tensor_to_boxes(model, img)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
    
        results.append(evaluator.evaluate_image(gt_boxes, boxes))
    
        # draw
        Draw_box(tensor_to_cv2(img), gt_boxes, temp_save_path.replace(".png", "_gt.png"), model_name=model_name)
        Draw_box(tensor_to_cv2(merge_image), boxes, temp_save_path, model_name=model_name)
    
    P, R, F = evaluator.combine_results(results)
    return P, R, F


def main(patch_path, save_path, model_name):
    pan_cfg = None
    config = "/home/dongli911/anaconda3/envs/udup/lib/python3.8/site-packages/mmocr/configs/textdet/"

    if model_name == "CRAFT":
        model = load_CRAFTmodel()
    elif model_name == "TCM":
        model = build_tcm_mmocr()
    elif model_name == "DBnet":
        from mmocr.utils.ocr import MMOCR  # 动态导入
        config = os.path.join(config, "dbnet/dbnet_r18_fpnc_1200e_icdar2015.py")
        model = MMOCR(det="DB_r18", det_config=config, recog=None)
    elif model_name == "EasyOCR":
        model = easyocr.Reader(["en"], gpu=True, model_storage_directory="AllConfig/all_model")
    elif model_name == "PanPP":
        model, pan_cfg = load_PANPlusmodel()
    elif model_name == "PanNet":
        from mmocr.utils.ocr import MMOCR  # 动态导入
        config = os.path.join(config, "panet/panet_r18_fpem_ffm_600e_icdar2015.py")
        model = MMOCR(det="PANet_IC15", det_config=config, recog=None)
    elif model_name == "PSENet":
        from mmocr.utils.ocr import MMOCR  # 动态导入
        config = os.path.join(config, "psenet/psenet_r50_fpnf_600e_icdar2015.py")
        model = MMOCR(det="PS_IC15", det_config=config, recog=None)
    elif model_name == "TextBoxesPP":
        weight_path = "AllConfig/all_model/ICDAR2013_TextBoxes.pth"
        tb_cls_thresh = 0.3
        tb_nms_thresh = 0.1
        net, encoder = load_TextBoxesPPmodel(weight_path, device="cuda:0")
        model = (net, encoder, tb_cls_thresh, tb_nms_thresh)
    elif model_name == "EAST":
        # 你可以通过环境变量 EAST_ROOT / EAST_CKPT 控制
        model = load_east_model(weight_path=EAST_CKPT, device="cuda:0")
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    if patch_path is not None:
        adv_patch1 = torch.load(patch_path).cuda()
        adv_patch2 = torch.load(patch_path.replace("advpatch1", "advpatch2")).cuda()
    else:
        adv_patch1 = None
        adv_patch2 = None
    
    P, R, F = evaluate_and_draw(model, adv_patch1, adv_patch2, save_path, pan_cfg, model_name=model_name)
    return P, R, F


def gen_iter_dict(root_eval, special_eval_item):
    mypath = os.path.join(root_eval, special_eval_item)
    dir_list = os.listdir(mypath)
    is_need = []
    for item in dir_list:
        if item.isdigit():
            is_need.append(item)
    return is_need


if __name__ == "__main__":
    model_list = ['CRAFT', 'DBnet', 'EasyOCR', 'PanPP', 'PanNet', 'PSENet', 'EAST', 'TextBoxesPP', 'TCM']
    mui_list = [0.09]
    iter_list = [27]  # [27, 34]对应0.09，0.12
    root_eval = os.path.join("results", "AllData_results", "results_uawup_eps100")
    special_eval = 'size=30_step=3_eps=100_lambdaw=0.01'
    sele_size = special_eval.split("=")[1].split("_")[0]
    # lambdaw = special_eval.split('=')[-1]
    lambdaw = special_eval.split("=")[4].split("_")[0]
    lambday = special_eval.split("=")[-1]

    for model_name in model_list:
        save_path_root = os.path.join("results/eval_adversarial/transfer", model_name)
        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)
        log_path = os.path.join(save_path_root, "log")
        log_file = open(log_path, "a")
    
        for mui, it in zip(mui_list, iter_list):
            patch_path = os.path.join(root_eval, special_eval, "advtorch", "advpatch1_{}".format(it))
            save_path = os.path.join(save_path_root, "{}_{}_{}".format(sele_size, it, mui))
    
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                for filename in os.listdir(save_path):
                    file_path = os.path.join(save_path, filename)
                    os.remove(file_path)
    
            P, R, F = main(patch_path, save_path, model_name)
            # logINFO("patch_size:{} lambdaw:{} iter:{} MUI:{} P:{:.4f} R:{:.4f} F:{:.4f}".format(sele_size, lambdaw, it, mui, P, R, F), log_file)
            logINFO("patch_size:{} lambdaw:{} lambday:{} iter:{} MUI:{} P:{:.4f} R:{:.4f} F:{:.4f}".format(
                    sele_size, lambdaw, lambday, it, mui, P, R, F), log_file)
    
        log_file.close()