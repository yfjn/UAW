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
from UAWUP.dataset import ImageDataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

evaluator=DetectionIoUEvaluator()

def logINFO(log, f):
    print(log, file=f, flush=True)
    print(log)

def evaluate_and_draw(model, adv_patch1, adv_patch2, image_root, gt_root, save_path, resize_ratio=0, is_resize=False, model_name="CRAFT"):
    global evaluator
    to_tensor = transforms.ToTensor()
    test_dataset = ImageDataset(transform=to_tensor, length=100, adv_patch_size=adv_patch1.shape, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    # image_names = [os.path.join(image_root, name) for name in os.listdir(image_root)]
    # images = [img_read(os.path.join(image_root, name)) for name in os.listdir(image_root)]
    # test_gts = [os.path.join(gt_root, name) for name in os.listdir(gt_root)]

    # Randomly select images
    # selected_indices = random.sample(range(len(image_names)), 10)
    # images = [images[i] for i in selected_indices]
    # image_names = [image_names[i] for i in selected_indices]
    # test_gts = [test_gts[i] for i in selected_indices]

    results = []  # PRF
    # for img, name, gt in tqdm(zip(images, image_names, test_gts)):
    for i, img in tqdm(enumerate(test_dataloader)):
        # temp_save_path = os.path.join(save_path, name.split('/')[-1])
        temp_save_path = os.path.join(save_path, '{:03d}.png'.format(i+1))
        h, w = img.shape[2:]
        if adv_patch1 != None:
            UAU = repeat_2patch_flip(adv_patch1.clone().detach(), adv_patch2.clone().detach(), h, w)
            mask_t = extract_background(img).to('cuda:0')
            img = img.to('cuda:0')
            merge_image = img * (1 - mask_t) + mask_t * UAU
        else:
            merge_image = img.clone().to('cuda:0')
        # cv2.imwrite(temp_save_path.replace('size_eps', 'size_eps_DU'), tensor_to_cv2(merge_image))
        merge_image = merge_image.to('cuda:0')
        if is_resize:
            merge_image = random_image_resize(merge_image, low=resize_ratio, high=resize_ratio)
            h, w = merge_image.shape[2:]
        if model_name == 'DBnet':
            preds = single_grad_inference(model, merge_image, [], model_name)
            preds = preds[0]
            dilates, boxes = get_DB_dilateds_boxes(preds, h, w, min_area=100)
        elif model_name == 'CRAFT':
            (score_text, score_link, target_ratio) = single_grad_inference(model, merge_image, [], model_name, is_eval=True)
            boxes = get_CRAFT_box(score_text, score_link, target_ratio, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
            (gt_score_text, gt_score_link, gt_target_ratio) = single_grad_inference(model, img, [], model_name, is_eval=True)
            gt_boxes = get_CRAFT_box(gt_score_text, gt_score_link, gt_target_ratio, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
        # gt = read_txt(gt)
        # results.append(evaluator.evaluate_image(gt, boxes))
        results.append(evaluator.evaluate_image(gt_boxes, boxes))
        # draw
        # cv2_img=cv2.imread(name)
        # Draw_box(cv2_img,np.array(gt),temp_save_path.replace('.png', '_gt.png'), model_name=model_name)
        # Draw_box(tensor_to_cv2(merge_image), boxes, temp_save_path, model_name=model_name)
        Draw_box(tensor_to_cv2(img), gt_boxes, temp_save_path.replace('.png', '_gt.png'), model_name=model_name)
        Draw_box(tensor_to_cv2(merge_image), boxes, temp_save_path, model_name=model_name)
    P, R, F = evaluator.combine_results(results)
    return P, R, F

def main(patch_path,img_path,gt_path,save_path,model_name):
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
    P, R, F = evaluate_and_draw(model,adv_patch1,adv_patch2, img_path,gt_path, save_path,model_name=model_name)
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
    model_name = "CRAFT"
    img_path = "AllData/test"  # 目前改成没用到
    gt_path = "AllData/test_craft_gt"  # 目前改成没用到
    root_eval = os.path.join('results', 'AllData_results', 'results_uawup_eps100')
    # special_eval = os.listdir(root_eval)
    special_eval=[
        # "size=10_step=3_eps=100_lambdaw=10",
        # "size=20_step=3_eps=100_lambdaw=1",
        "size=30_step=3_eps=100_lambdaw=0.01",
        # "size=40_step=3_eps=100_lambdaw=0.001",
        # "size=50_step=3_eps=100_lambdaw=0.01"
    ]
    save_path_root = os.path.join("eval_adversarial", root_eval.split('/')[-1], "size_eps")  # 添加DU与否的检测结果
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    log_path = os.path.join(save_path_root, 'log')
    log_file = open(log_path, 'a')
    
    for dir in special_eval:
        sele_size=dir.split('=')[1].split('_')[0]
        lambdaw=dir.split('=')[-1]
        iter_eval=gen_iter_dict(root_eval,dir)
        iter_eval=sorted(iter_eval,key=lambda x:int(x))
        iter_eval=iter_eval[3:4]  # MUI=0.09
        # iter_eval=iter_eval[4:5]  # MUI=0.1
        for iter in iter_eval:
            patch_path = os.path.join(root_eval,dir,"advtorch","advpatch1_{}".format(iter))
            save_path = os.path.join(save_path_root, '{}_{}_{}'.format(sele_size, lambdaw, iter))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                for filename in os.listdir(save_path):
                    file_path = os.path.join(save_path, filename)
                    os.remove(file_path)
    
            P, R, F = main(patch_path, img_path, gt_path, save_path, model_name)
            logINFO("patch_size:{} lambdaw:{} iter:{} P:{:.4f} R:{:.4f} F:{:.4f}".format(sele_size, lambdaw, iter, P, R, F), log_file)
            logINFO("{:.4f}/{:.4f}/{:.4f}".format(R, P, F), log_file)
