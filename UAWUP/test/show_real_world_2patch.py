import sys
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/PycharmProjects/UAWUP")
from AllConfig.GConfig import abspath
import os
from Tools.CRAFTTools import *
from Tools.ImageIO import *
arabic_path=os.path.join(abspath,"AllData/real-world/arabic.png")
chinese_path=os.path.join(abspath,"AllData/real-world/chinese.png")
english_path=os.path.join(abspath,"AllData/real-world/english.png")
japanese_path=os.path.join(abspath,"AllData/real-world/japanese.png")
#x_start,x_end,y_start,y_end
arabic_range=[0,1317,180,1085]
chinese_range=[0,1200,200,1283]
english_range=[0,850,120,1095]
japanese_range=[0,920,10,1253]

adv_patch_path='results/result_uawup/size=30_step=3_eps=120_lambdaw=0.1/advtorch/advpatch1_27'
# adv_patch_path='results/results_uawup/size=30_step=3_eps=120_lambdaw=0.1/advtorch/advpatch1_30'
save_path=os.path.join(abspath,"test","real_world")
os.makedirs(save_path,exist_ok=True)
CRAFTnet=load_CRAFTmodel()

def extract_background2(img_tensor: torch.Tensor):
    assert (len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1)
    img_sum = torch.sum(img_tensor, dim=1)
    mask = (img_sum >2.9)
    mask = mask + 0
    return mask.unsqueeze_(0)

def pred_path(img,save_name):
    global save_path
    score_text, score_link, target_ratio = get_CRAFT_pred(CRAFTnet, img=img, square_size=1280, is_eval=True)
    boxes = get_CRAFT_box(score_text, score_link, target_ratio,
                          text_threshold=0.7, link_threshold=0.4, low_text=0.4)
    # 保存结果到cv图片
    img = tensor_to_cv2(img)
    CRAFT_draw_box(img, boxes=boxes, save_path=os.path.join(save_path,save_name))

def gen_uimg(img_path,range_path,adv_patch_path):
    adv_patch1=torch.load(adv_patch_path)
    adv_patch2=torch.load(adv_patch_path.replace('advpatch1','advpatch2'))

    img=img_read(img_path)
    h, w = img.shape[2:]
    UAU = repeat_2patch(adv_patch1.clone().detach(), adv_patch2.clone().detach(), h, w).cuda()
    mask_t = extract_background2(img).cuda()
    #给不需要的地方赋值0
    mask_2=torch.zeros_like(mask_t)
    mask_2[:,:,range_path[-2]:range_path[-1],range_path[0]:range_path[1]]=1
    mask_2=mask_2.cuda()
    mask_t=mask_2*mask_t

    img = img.cuda()
    merge_image = img * (1 - mask_t) + mask_t * UAU
    merge_image = merge_image.cuda()
    return merge_image

img=gen_uimg(english_path,english_range,adv_patch_path)

pred_path(img=img,save_name="english.png")