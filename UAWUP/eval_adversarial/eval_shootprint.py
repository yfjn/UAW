# 测试一对底纹的对抗性
# 测试打印捕获的STD对抗性
import sys
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
import warnings
warnings.filterwarnings("ignore")

evaluator=DetectionIoUEvaluator()

def evaluate_and_draw(model,adv_patch, image_root, gt_root,
                      save_path, resize_ratio=0, is_resize=False,
                      model_name="CRAFT"):
    global evaluator
    image_names = [os.path.join(image_root, name) for name in os.listdir(image_root)]
    images = [img_read(os.path.join(image_root, name)) for name in os.listdir(image_root)]
    # test_gts = [os.path.join(gt_root, name) for name in os.listdir(gt_root)]

    # # Randomly select images
    # selected_indices = random.sample(range(len(image_names)), 70)
    # images = [images[i] for i in selected_indices]
    # image_names = [image_names[i] for i in selected_indices]
    # test_gts = [test_gts[i] for i in selected_indices]

    results = []  # PRF
    for img, name in tqdm(zip(images, image_names)):
        temp_save_path = os.path.join(save_path, name.split('/')[-1])
        h, w = img.shape[2:]
        if adv_patch != None:
            UAU = repeat_4D(adv_patch.clone().detach(), h, w)
            mask_t = extract_background(img).to('cuda:0')
            img = img.to('cuda:0')
            merge_image = img * (1 - mask_t) + mask_t * UAU
        else:
            merge_image = img.clone().to('cuda:0')
        # cv2.imwrite(temp_save_path.replace('eval_patch', 'eval_patch_DU'), tensor_to_cv2(merge_image))
        merge_image = merge_image.to('cuda:0')
        if is_resize:
            merge_image = random_image_resize(merge_image, low=resize_ratio, high=resize_ratio)
            h, w = merge_image.shape[2:]
        if model_name == 'DBnet':
            preds = single_grad_inference(model, merge_image, [], model_name)
            preds = preds[0]
            dilates, boxes = get_DB_dilateds_boxes(preds, h, w, min_area=100)
        elif model_name == 'CRAFT':
            (score_text, score_link, target_ratio) = single_grad_inference(model, merge_image, [],
                                                                           model_name)
            boxes = get_CRAFT_box(score_text, score_link, target_ratio,
                                  text_threshold=0.7, link_threshold=0.4, low_text=0.4)
        # gt = read_txt(gt)
        # results.append(evaluator.evaluate_image(gt, boxes))
        # draw
        # cv2_img=cv2.imread(name)
        # Draw_box(cv2_img,np.array(gt),temp_save_path.replace('.png', '_gt.png'))
        Draw_box(tensor_to_cv2(merge_image), boxes, temp_save_path, model_name=model_name)
    # P, R, F = evaluator.combine_results(results)
    # return P, R, F


def main(patch_path,img_path,gt_path,save_path,model_name):
    if model_name=='CRAFT':
        model = load_CRAFTmodel()
    else:
        model = load_DBmodel()
    if patch_path != None:
        adv_patch = torch.load(patch_path)
        adv_patch = adv_patch.cuda()
    else:
        adv_patch = None
    evaluate_and_draw(model,adv_patch, img_path,gt_path, save_path,model_name=model_name)
    # print("R:{},P:{},F:{}".format(R,P,F))
    # return R, P, F


def logINFO(log, f):
    print(log, file=f, flush=True)
    print(log)


if __name__=="__main__":
    model_name = "CRAFT"
    img_path_root_root = "../testimg/shootprint_crop_cc_"
    save_path_root = "eval_adversarial/result/shootprint/"
    w_list = ["100w/", "150w/", "200w/"]
    for w in w_list:
        img_path_root = img_path_root_root + w
        print("img_path_root:", img_path_root)
        img_path = [os.path.join(img_path_root,folder) for folder in os.listdir(img_path_root)]
        print(img_path)
        for path in img_path:
            save_path = save_path_root + "shootprint_crop_cc_" + w + path.split('/')[-1]
            print("save_path:", save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            main(None, path, None, save_path, model_name)

    # for s in list:
    #     if s != 0:
    #         patch_path = patch_path_root + '{}.pth'.format(s)
    #     else:
    #         patch_path = None
    #     save_path = save_path_root + '{}'.format(s)
    #     save_path_DU = save_path_DU_root + '{}'.format(s)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     else:
    #         for filename in os.listdir(save_path):
    #             file_path = os.path.join(save_path, filename)
    #             os.remove(file_path)
    #     if not os.path.exists(save_path_DU):
    #         os.makedirs(save_path_DU)
    #     else:
    #         for filename in os.listdir(save_path_DU):
    #             file_path = os.path.join(save_path_DU, filename)
    #             os.remove(file_path)
    #     # print("patch_path:", patch_path)
    #     # print("save_path:", save_path)
    
    #     R, P, F = main(patch_path, img_path, gt_path, save_path, model_name)
    #     logINFO("patch_size:{} R:{:.4f} P:{:.4f} F:{:.4f}".format(s, R, P, F), log_file)
