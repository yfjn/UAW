# resize_jpeg 测试
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
from watermarklab.noiselayers.testdistortions import *
from tqdm import tqdm
from dataset import ImageDataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

evaluator=DetectionIoUEvaluator()

def logINFO(log, f):
    print(log, file=f, flush=True)
    print(log)

def evaluate_and_draw(model, adv_patch1, adv_patch2, image_root, distortion, f, save_path, resize_ratio=0, model_name="CRAFT"):
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
        if distortion:
            merge_image = distortion.test(tensor_to_cv2(merge_image), tensor_to_cv2(img), f)
            merge_image = transforms.ToTensor()(merge_image).unsqueeze(0)
            h, w = merge_image.shape[2:]
            img = transforms.Resize([h, w])(img)
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
    return P, R, F

def main(patch_path,img_path,distortion,f,save_path,model_name):
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
    P, R, F = evaluate_and_draw(model,adv_patch1,adv_patch2,img_path,distortion,f,save_path,model_name=model_name)
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
    model_name = "CRAFT"
    img_path = "AllData/test"
    root_eval = os.path.join('results', 'AllData_results', 'results_uawup_eps100')
    special_eval=[
        "size=30_step=3_eps=100_lambdaw=0.01"
    ]
    noises=[
        "Identity",
        "Brightness",
        "Contrast",
        "Saturation",
        "Hue",
        "Gaussian Noise",
        "Gaussian Blur",
        "Jpeg",
        "Resize",
        "Crop",
        "Perspective Transformation"
    ]
    
    for dir in special_eval:
        sele_size=dir.split('=')[1].split('_')[0]
        iter_eval=gen_iter_dict(root_eval, dir)
        iter_eval=sorted(iter_eval,key=lambda x:int(x))
        iter_eval=iter_eval[3:4]  # MUI=0.09
        # iter_eval=iter_eval[4:5]  # MUI=0.1
        for iter in iter_eval:
            for noisename in noises:
                if noisename == 'Identity':
                    distortion = Identity()
                    factors = [1.]
                if noisename == 'Brightness':
                    distortion = Brightness()
                    factors = [0.85, 1.15]
                elif noisename == 'Contrast':
                    distortion = Contrast()
                    factors = [0.85, 1.15]
                elif noisename == 'Saturation':
                    distortion = Saturation()
                    factors = [0.85, 1.15]
                elif noisename == 'Hue':
                    distortion = Hue()
                    factors = [-0.1, 0.1]
                elif noisename == 'Gaussian Noise':
                    distortion = GaussianNoise()
                    factors = [0.05, 0.1]
                elif noisename == 'Gaussian Blur':
                    distortion = GaussianBlur()
                    factors = [0.5, 1.0]
                elif noisename == 'Jpeg':
                    distortion = Jpeg()
                    factors = [90, 70, 50]
                elif noisename == 'Resize':
                    distortion = Resize()
                    factors = [0.8, 1.5]
                elif noisename == 'Crop':
                    distortion = Crop()
                    factors = [0.6, 0.8]
                # 旋转和翻转没意义
                elif noisename == 'Perspective Transformation':
                    distortion = RandomCompensateTransformer()
                    factors = [4, 8]
                save_path_root = os.path.join("eval_adversarial", root_eval.split('/')[-1], noisename)
                if not os.path.exists(save_path_root):
                    os.makedirs(save_path_root)
                log_path = os.path.join(save_path_root, 'log')
                log_file = open(log_path, 'a')
                for f in factors:
                    patch_path = os.path.join(root_eval,dir,"advtorch","advpatch1_{}".format(iter))
                    save_path = os.path.join(save_path_root, '{}_{}_{}'.format(sele_size, iter, f))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    else:
                        for filename in os.listdir(save_path):
                            file_path = os.path.join(save_path, filename)
                            os.remove(file_path)
            
                    P, R, F = main(patch_path, img_path, distortion, f, save_path, model_name)
                    logINFO("patch_size:{} iter:{} noisename:{} factor:{} P:{:.4f} R:{:.4f} F:{:.4f}".format(sele_size, iter, noisename, f, P, R, F), log_file)
