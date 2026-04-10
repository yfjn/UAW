import sys
import time

from dataset import ImageDataset
from torch.utils.data import DataLoader
sys.path.append("..")
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
# from model_DBnet.pred_single import *
import torch.nn.functional as F
import AllConfig.GConfig as GConfig
from Tools.ImageIO import img_read, tensor_to_cv2
from Tools.Baseimagetool import *
import random
from UDUP.Auxiliary import *
from Tools.ImageIO import logger_config
import datetime
from Tools.EvalTool import DetectionIoUEvaluator,read_txt
from Tools.PIAttack import project_noise
from mmocr.utils.ocr import MMOCR
from PIL import Image
import numpy as np
from torchvision import transforms
import torchattacks
import easyocr
from model_CRAFT.pred_single import *
from model_DBnet.pred_single import *
from model_PAN.pred_single import pan_pred_single
from Tools.PANTools import *
from Tools.CRAFTTools import *
from Tools.DBTools import *

to_tensor = transforms.ToTensor()


class RepeatAdvPatch_Attack():
    def __init__(self,
        data_root,
        savedir,
        log_name,
        save_mui,
        eps,
        alpha,
        decay,
        T,
        batch_size,
        lm_mui_thre=0.06,
        adv_patch_size=(1, 3, 30, 30),
        gap=5,
        lambdaw=0.01,
        lambdax=0,
        model_name="DBnet",
        evaluate=True,
        gray=True,
        debug=False
    ):
        random.seed()  # 让 Python random 回到系统默认
        torch.manual_seed(torch.initial_seed())  # 让 torch 继续随机
        np.random.seed()  # 让 numpy 继续随机
        self.model_name=model_name

        # 加载指定的模型
        if model_name == 'CRAFT':
            self.CRAFTmodel = load_CRAFTmodel()
            self.CRAFTmodel_2 = load_CRAFTmodel()
            # CRAFT 中间层: basenet.slice2 提取, upconv2/upconv1 上采样, conv_cls 预测文字概率图
            self.midLayer = ["basenet.slice2.14", "basenet.slice2.17", "upconv2.conv.3", "upconv1.conv.3", "conv_cls.8"]
        elif model_name == 'EasyOCR':
            self.EasyOCR_model = easyocr.Reader(['en'], gpu=True, model_storage_directory='AllConfig/all_model')
            self.EasyOCR_model_2 = easyocr.Reader(['en'], gpu=True, model_storage_directory='AllConfig/all_model')
            # EasyOCR-CRAFT 中间层: basenet.slice2/3/4/5 提取, conv_cls 预测文字概率图
            # 注意: EasyOCR 使用的 CRAFT detector 结构与原版相同，最后一层也是 conv_cls
            self.midLayer = ["basenet.slice5", "basenet.slice4", "basenet.slice3", "basenet.slice2", "conv_cls.8"]
        else:
            raise ValueError(f"Unsupported model: {model_name}. Only CRAFT and EasyOCR are supported.")

        # self.midLayer = ["bbox_head.binarize.7"]
        # hyper-parameters
        self.eps, self.alpha, self.decay = eps, alpha, decay

        # train settings
        self.T = T
        self.batch_size = batch_size
        self.eval=evaluate  # 原来没用上，我改成表示是否固定STD参数
        self.gray = gray
        self.debug = debug

        # Loss
        self.Mseloss = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.lambdaw= lambdaw
        self.lambdax= lambdax
        self.lm_mui_thre=lm_mui_thre

        # path process
        self.data_root = data_root
        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.makedirs(self.savedir)
        self.train_dataset = init_train_dataset(data_root)
        self.test_dataset = init_test_dataset(data_root)
        # train_dataset = ImageDataset(transform=to_tensor, length=batch_size, adv_patch_size=adv_patch_size, test=False)
        # self.train_dataset = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
        # test_dataset = ImageDataset(transform=to_tensor, length=50, adv_patch_size=adv_patch_size, test=True)
        # self.test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        # all gap
        self.shufflegap = len(self.train_dataset) // self.batch_size
        self.gap = gap

        # initiation
        self.adv_patch1 = torch.ones(list(adv_patch_size))
        self.adv_patch2 = torch.ones(list(adv_patch_size))
        self.start_epoch = 1
        self.t=0
        # recover adv patch
        recover_adv_path1, recover_adv_path2, recover_t = self.recover_adv_patch()
        if recover_t != None:
            self.adv_patch1 = recover_adv_path1
            self.adv_patch2 = recover_adv_path2
            self.t = recover_t


        self.logger = logger_config(log_filename=log_name)
        while len(self.logger.handlers)!=0:
            self.logger.removeHandler(self.logger.handlers[0])
        self.logger = logger_config(log_filename=log_name)
        
        eps_display = f"{eps:.6f}"
        alpha_display = f"{alpha:.6f}"
        self.logger.info(f'model: {model_name}, size: {adv_patch_size[-1]}, eps: {eps_display}, step: {alpha_display}, lambdaw: {lambdaw}, lambday: {lambdax}')
        self.evaluator=DetectionIoUEvaluator()

        self.save_mui=save_mui
        self.save_mui_flag=[0 for _ in range(len(save_mui))]  #0 是没保存，1是保存
        self.train_start_time = time.time()


    def is_need_save(self,mui_now,gap=0.002):  # 返回0，不需要保存，返回1需要保存，返回2直接退出
        save_array=np.array(self.save_mui)  # 需要保存是true
        save_array=save_array>mui_now
        save_index=np.sum(save_array==False)
        
        if save_index>0 and self.save_mui_flag[save_index-1]!=1:  # 如果超出了保存值，但是又没保存，即立刻保存
            self.save_mui_flag[save_index-1] = 1
            return 1
        if save_index==len(self.save_mui):
            return 2
        mui_gap=self.save_mui[save_index]-mui_now
        if mui_gap<=gap and self.save_mui_flag[save_index]!=1:
            self.save_mui_flag[save_index]=1
            return 1
        return 0

    def get_model_by_name(self, model_name, second=False):
        """
        Get model instance by name
        """
        if model_name == 'CRAFT':
            return self.CRAFTmodel_2 if second else self.CRAFTmodel
        elif model_name == 'EasyOCR':
            return self.EasyOCR_model_2 if second else self.EasyOCR_model
        else:
            raise ValueError(f"Unsupported model: {model_name}. Only CRAFT and EasyOCR are supported.")

    def recover_adv_patch(self):
        temp_save_path = os.path.join(self.savedir, "advtorch")
        if os.path.exists(temp_save_path):
            files = os.listdir(temp_save_path)
            if len(files) == 0:  # 如果目录存在且不为空，按照文件名中的时间步排序
                return None, None, None
            files = sorted(files, key=lambda x: int(x.split('.')[0].split("_")[-1]))
            t = int(files[-1].split("_")[-1])  # 加载最新保存的对抗性补丁文件，并返回补丁和对应的时间步t
            keyfile2 = os.path.join(temp_save_path, files[-1])
            keyfile1 = os.path.join(temp_save_path, files[-1].replace('advpatch2', 'advpatch1'))
            return torch.load(keyfile1), torch.load(keyfile2), t
        return None, None, None

    #self.CallLoss(res, res_du, x_d2 - DU_d2, it_adv_patch, m_d2)
    def CallLoss(self,text_feamap,uau_feamap,diff,adv_patch1,adv_patch2,modle_name,now_p, debug=False, t=0):
        # 处理最后一层特征作为预测文字概率图
        pred = text_feamap[-1]
        
        # 根据不同的模型设置目标
        # if modle_name=='CRAFT':
            # CRAFT: 目标是降低文字检测得分
        target = torch.ones_like(pred)
        target = target * (-0.1)
        # else:
        #     # 其他模型（EasyOCR/DBnet/PanPP/PanNet/PSENet）: 目标是零化
        #     target = torch.zeros_like(pred)
        
        target = target.to(pred.device)
        dloss = self.Mseloss(pred, target)
        log_dloss = dloss.detach().cpu().item()

        # 计算多层特征匹配损失
        mmloss = 0
        if len(text_feamap) > 1 and len(uau_feamap) > 1:
            feature_loss_count = 0  # 计算中间层特征的损失
            for f1, f2 in zip(text_feamap[:-1], uau_feamap[:-1]):
                mmloss += self.Mseloss(f1.mean([2, 3]), f2.mean([2, 3]))
                feature_loss_count += 1
            
            diff_norm = torch.norm(diff, p=1)  # 计算L1范数用于标准化
            diff_norm = torch.clamp(diff_norm, min=1e-8)
            mmloss = mmloss / diff_norm
        else:
            mmloss = self.Mseloss(text_feamap[-1], uau_feamap[-1])  # 如果只有一层特征，使用预测特征直接计算
        
        log_mmloss = mmloss.detach().cpu().item() if isinstance(mmloss, torch.Tensor) else mmloss

        # 计算补丁一致性损失
        l1loss = self.Mseloss(adv_patch1, adv_patch2)
        log_l1loss = l1loss.detach().cpu().item()

        # 组合总损失
        if now_p > self.lm_mui_thre:
            total_loss = dloss + mmloss * self.lambdaw - l1loss * self.lambdax
        else:
            total_loss = dloss - l1loss * self.lambdax
        
        # 检查 NaN
        if torch.isnan(total_loss):
            self.logger.error(f"ERROR: total_loss is NaN! dloss={log_dloss}, mmloss={log_mmloss}, l1loss={log_l1loss}")
            total_loss = dloss
        
        # 计算梯度（保留对补丁的梯度）
        grad1, grad2 = torch.autograd.grad(total_loss, [adv_patch1, adv_patch2], retain_graph=False, create_graph=False, allow_unused=True)
        
        # 处理 None 梯度
        if grad1 is None:
            grad1 = torch.zeros_like(adv_patch1)
            self.logger.warning(f"WARNING: grad1 is None for model {modle_name}")
        if grad2 is None:
            grad2 = torch.zeros_like(adv_patch2)
            self.logger.warning(f"WARNING: grad2 is None for model {modle_name}")
        
        grad1 = grad1.detach().cpu()
        grad2 = grad2.detach().cpu()
        
        return grad1, grad2, log_dloss, log_mmloss, log_l1loss

    # 快捷初始化
    def inner_init_adv_patch_image(self, mask, image, hw, device):
        adv_patch = self.adv_patch.clone().detach()
        adv_patch = adv_patch.to(device)
        adv_patch.requires_grad = True
        image = image.to(device)
        adv_image = self.get_merge_image(adv_patch, mask=mask, image=image, hw=hw, device=device)
        return adv_patch, adv_image

    def train(self):
        momentum1 = 0
        momentum2 = 0
        # alpha_beta =  5/255
        # gamma = alpha_beta
        # amplification = 0.0

        self.logger.info("start optimizing=====================")
        shuff_ti = 0  # train_dataset_iter
        for t in range(self.t + 1, self.T):
            if t % self.shufflegap == 0:
                random.shuffle(self.train_dataset)
                shuff_ti = 0
            batch_dataset = self.train_dataset[shuff_ti * self.batch_size: (shuff_ti + 1) * self.batch_size]
            shuff_ti += 1

            batch_dLoss = 0
            batch_mmLoss = 0
            batch_l1loss = 0
            sum_grad1 = torch.zeros_like(self.adv_patch1)
            sum_grad2 = torch.zeros_like(self.adv_patch2)

            now_p=torch.mean(torch.ones_like(self.adv_patch1) - self.adv_patch1.cpu())
            now_p+=torch.mean(torch.ones_like(self.adv_patch2) - self.adv_patch2.cpu())
            now_p*=0.5

            for i, [x] in enumerate(batch_dataset):
            # for i, x in enumerate(self.train_dataset):
                it_adv_patch1=self.adv_patch1.clone().detach().to('cuda:0')
                it_adv_patch2=self.adv_patch2.clone().detach().to('cuda:0')
                it_adv_patch1.requires_grad=True
                it_adv_patch2.requires_grad=True
                x = x.to('cuda:0')
                x_d1 = Diverse_module_1(x, t, self.gap)
                # x_d1 = x
                m = extract_background(x_d1)  # character region
                h, w = x_d1.shape[2:]
                DU = repeat_2patch(patch1=it_adv_patch1, patch2=it_adv_patch2, h_real=h, w_real=w)
                merge_x = DU * m + x_d1 * (1 - m)
                x_d2, DU_d2 = Diverse_module_2(image=merge_x,UAU=DU,now_ti=t, gap=self.gap)
                # x_d2, DU_d2 = Diverse_module_adv(image=merge_x,x_d1=x_d1,UAU=DU,now_ti=t, gap=self.gap)

                # Get features using single_grad_inference for both CRAFT and EasyOCR
                model = self.get_model_by_name(self.model_name, second=False)
                model_2 = self.get_model_by_name(self.model_name, second=True)
                
                _, res = single_grad_inference(model, x_d2, self.midLayer, self.model_name, is_eval=self.eval)
                _, res_du = single_grad_inference(model_2, DU_d2, self.midLayer, self.model_name, is_eval=self.eval)
                res = [v.fea.clone() for v in res]
                res_du = [v.fea.clone() for v in res_du]

                # DEBUG: Only output detailed debug for first iteration of each epoch (every 10 iterations)
                debug_iter = self.debug and (t % 10 == 0) and i == 0
                grad1,grad2,temp_dloss,temp_mmloss,temp_l1loss=self.CallLoss(res,res_du,x_d2-DU_d2,it_adv_patch1,it_adv_patch2,self.model_name,now_p.item(), debug=debug_iter, t=t)
                sum_grad1 += grad1
                sum_grad2 += grad2
                batch_dLoss += temp_dloss
                # print(i, temp_mmloss)  # Diverse_module_2不加均匀噪声mmloss可能会出现nan，不知道为什么
                batch_mmLoss += temp_mmloss
                batch_l1loss += temp_l1loss
                torch.cuda.empty_cache()
                
            temp_save_path = os.path.join(self.savedir, "noise_img")
            if os.path.exists(temp_save_path) == False:
                os.makedirs(temp_save_path)
            save_adv_patch_img(x_d2, os.path.join(temp_save_path, "x_d2_{}.png".format(t)))
            save_adv_patch_img(DU_d2, os.path.join(temp_save_path, "DU_d2_{}.png".format(t)))

            # update grad，安全的梯度归一化，避免 NaN
            abs_sum_grad1 = torch.mean(torch.abs(sum_grad1), dim=(1), keepdim=True)
            abs_sum_grad1 = torch.clamp(abs_sum_grad1, min=1e-8)
            grad1 = sum_grad1 / abs_sum_grad1
            
            abs_sum_grad2 = torch.mean(torch.abs(sum_grad2), dim=(1), keepdim=True)
            abs_sum_grad2 = torch.clamp(abs_sum_grad2, min=1e-8)
            grad2 = sum_grad2 / abs_sum_grad2
            
            grad1 = grad1 + momentum1 * self.decay
            grad2 = grad2 + momentum2 * self.decay
            momentum1 = grad1
            momentum2 = grad2

            # update adv_patch
            temp_patch1 = self.adv_patch1.clone().detach().cpu() - self.alpha * grad1.sign()
            temp_patch1 = torch.clamp(temp_patch1, min=1-self.eps, max=1)
            self.adv_patch1 = temp_patch1
            temp_patch2 = self.adv_patch2.clone().detach().cpu() - self.alpha * grad2.sign()
            temp_patch2 = torch.clamp(temp_patch2, min=1-self.eps, max=1)
            self.adv_patch2 = temp_patch2

            if self.gray:
                gray_patch1 = self.adv_patch1.mean(dim=1, keepdim=True)  # (1, 1, H, W)
                self.adv_patch1 = gray_patch1.expand(-1, 3, -1, -1).clone()  # (1, 3, H, W)
                gray_patch2 = self.adv_patch2.mean(dim=1, keepdim=True)
                self.adv_patch2 = gray_patch2.expand(-1, 3, -1, -1).clone()

            # update logger
            now_mui=torch.mean(torch.ones_like(temp_patch1) - temp_patch1.cpu())
            now_mui+=torch.mean(torch.ones_like(temp_patch2) - temp_patch2.cpu())
            now_mui*=0.5
            
            # Format losses with 4 significant figures for display
            dloss_display = f"{batch_dLoss/self.batch_size:.6f}"
            mmloss_display = f"{batch_mmLoss/self.batch_size:.6f}"
            l1loss_display = f"{batch_l1loss/self.batch_size:.6f}"
            pert_display = f"{now_mui:.6f}"
            e="iter:{}, dloss:{}, mmLoss:{}, l1loss:{}, pert:{}".format(t, dloss_display, mmloss_display, l1loss_display, pert_display)
            self.logger.info(e)

            # save adv_patch with
            temp_save_path = os.path.join(self.savedir, "advpatch")
            if os.path.exists(temp_save_path) == False:
                os.makedirs(temp_save_path)
            save_adv_patch_img(self.adv_patch1, os.path.join(temp_save_path, "advpatch1_{}.png".format(t)))
            save_adv_patch_img(self.adv_patch2, os.path.join(temp_save_path, "advpatch2_{}.png".format(t)))
            temp_torch_save_path = os.path.join(self.savedir, "advtorch")
            if os.path.exists(temp_torch_save_path) == False:
                os.makedirs(temp_torch_save_path)
            torch.save(self.adv_patch1, os.path.join(temp_torch_save_path, "advpatch1_{}".format(t)))
            torch.save(self.adv_patch2, os.path.join(temp_torch_save_path, "advpatch2_{}".format(t)))

            #根据save_mui进行测试
            is_save=self.is_need_save(now_mui.item())
            if is_save==1:
                self.evaluate_test_path(t)
            elif is_save==2:
                self.logger.info("OVER=====================")
                break


    def evaluate_and_draw(self,adv_patch1,adv_patch2,image_root,gt_root,save_path,resize_ratio=0,is_resize=False):
        image_names = [os.path.join(image_root, name) for name in os.listdir(image_root)]
        images = [img_read(os.path.join(image_root, name)) for name in os.listdir(image_root)]
        test_gts = [os.path.join(gt_root, name) for name in os.listdir(gt_root)]
        results=[]  # PRF
        for img,name,gt in zip(images,image_names,test_gts):
        # for i, img in enumerate(self.test_dataset):
            h,w=img.shape[2:]
            UAU=repeat_2patch(adv_patch1.clone().detach(),adv_patch2.clone().detach(),h,w)
            # UAU=repeat_2patch_flip(adv_patch1.clone().detach(),adv_patch2.clone().detach(),h,w)
            mask_t=extract_background(img)
            merge_image=img*(1-mask_t)+mask_t*UAU
            merge_image=merge_image.to('cuda:0')
            if is_resize:
                merge_image=random_image_resize(merge_image,low=resize_ratio,high=resize_ratio)
                h, w = merge_image.shape[2:]

            # Get boxes based on model type
            model = self.get_model_by_name(self.model_name)
            if self.model_name == 'CRAFT':
                (score_text, score_link, target_ratio) = single_grad_inference(model, merge_image, [], self.model_name, is_eval=True)
                boxes = get_CRAFT_box(score_text, score_link, target_ratio, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
            elif self.model_name == 'EasyOCR':
                tmp_merge_image = tensor_to_cv2(merge_image)
                result = model.readtext(tmp_merge_image)
                boxes = np.array([item[0] for item in result])

            gt=read_txt(gt)
            results.append(self.evaluator.evaluate_image(gt,boxes))
            # results.append(self.evaluator.evaluate_image(gt_boxes,boxes))
            #draw
            cv2_img=cv2.imread(name)
            name=name.split(os.sep)[-1]
            Draw_box(cv2_img,np.array(gt),os.path.join(save_path,name.split('.')[0]+'_gt.png'),model_name=self.model_name)
            # Draw_box(tensor_to_cv2(img), gt_boxes, os.path.join(save_path, '{:03d}_gt.png'.format(i)), model_name=self.model_name)
            temp_save_path=os.path.join(save_path,name)
            Draw_box(tensor_to_cv2(merge_image), boxes, temp_save_path,model_name=self.model_name)
            # Draw_box(tensor_to_cv2(merge_image), boxes, os.path.join(save_path, '{:03d}.png'.format(i)), model_name=self.model_name)
        P, R, F = self.evaluator.combine_results(results)
        return P,R,F

    def evaluate_test_path(self, t, eval=True, eval_scale=False):
        elapsed_time = time.time() - self.train_start_time
        self.logger.info(f"[Timing] Bit Templates Generation optimization elapsed time: {elapsed_time:.2f} seconds")

        if eval:
            o_img_root=os.path.join(self.data_root,'test')
            o_gt_root = os.path.join(self.data_root, 'test_craft_gt')
            o_save_dir = os.path.join(self.savedir, str(t),'original')
            if os.path.exists(o_save_dir) == False:
                os.makedirs(o_save_dir)
            P,R,F=self.evaluate_and_draw(self.adv_patch1,self.adv_patch2,o_img_root,o_gt_root,o_save_dir)
            # Format evaluation metrics with 4 significant figures
            p_display = f"{P:.6f}"
            r_display = f"{R:.6f}"
            f_display = f"{F:.6f}"
            e="iter:{}, original--P:{}, R:{}, F:{}".format(t, p_display, r_display, f_display)
            self.logger.info(e)

        #data_root
            #test_resize
            #test_resize_gt
                #60
                #...
                #200
        if eval_scale:
            resize_scales = [item / 10 for item in range(6, 21, 1)]  # 0.6 0.7 0.8 ... 2.0
            for item in resize_scales:
                str_s=str(int(item * 100))
                r_img_root=os.path.join(self.data_root,'test_resize')
                r_gt_root = os.path.join(self.data_root, 'test_resize_gt',str_s)
                r_save_dir = os.path.join(self.savedir, str(t), str_s)
                if os.path.exists(r_save_dir) == False:
                    os.makedirs(r_save_dir)
                P,R,F=self.evaluate_and_draw(self.adv_patch,r_img_root,r_gt_root,r_save_dir)
                e="iter:{},scale_ratio:{},P:{},R:{},F:{}".format(t,item,P,R,F)
                self.logger.info(e)
