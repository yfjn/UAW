import sys
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/PycharmProjects/UAW")
import cv2
import numpy as np
from tqdm import tqdm
import os
from watermarklab.noiselayers.testdistortions import *

def gen_iter_dict(root_eval,special_eval_item):
    mypath=os.path.join(root_eval,special_eval_item)
    dir_list=os.listdir(mypath)
    is_need=[]
    for item in dir_list:
        if item.isdigit():
            is_need.append(item)
    return is_need

if __name__ == "__main__":
    # 输入文件路径和输出路径
    root_eval = os.path.join('results', 'AllData_results', 'results_uawup_eps100')
    special_eval=[
        # "size=10_step=3_eps=100_lambdaw=10",
        # "size=20_step=3_eps=100_lambdaw=1",
        "size=30_step=3_eps=100_lambdaw=0.01",
        # "size=40_step=3_eps=100_lambdaw=0.001",
        # "size=50_step=3_eps=100_lambdaw=0.01"
    ]
    noises={
        "Identity": [1.],
        "Brightness": [0.7, 0.85, 1.15, 1.3],
        "Contrast": [0.7, 0.85, 1.15, 1.3],
        "Saturation": [0.7, 0.85, 1.15, 1.3],
        "Hue": [-0.2, -0.1, 0.1, 0.2],
        "Gaussian Noise": [0.05, 0.1, 0.15, 0.2],
        "Gaussian Blur": [1.5, 2.0],  # 0.5, 1.0, 
        "Jpeg": [90, 70, 50, 30, 10],
        "Resize": [0.6, 1.2],  # (h,w): (0.6,0.7), (0.8,0.9), (1.2,1.1), (1.5,1.4), 0.8, 1.5
        "Crop": [0.6, 0.8],
        "Rotate": [90, 180, 270],
        "Flip": [0, 1, -1],
    }

    for dir in special_eval:
        sele_size=dir.split('=')[1].split('_')[0]
        iter_eval=gen_iter_dict(root_eval, dir)
        iter_eval=sorted(iter_eval,key=lambda x:int(x))
        iter_eval=iter_eval[3:4]  # MUI=0.09
        # iter_eval=iter_eval[4:5]  # MUI=0.1

        for iter in iter_eval:
            input_dir = "eval_watermark/size_it/awti_{}_{}".format(sele_size, iter)
            output_dir = "eval_watermark/noise"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for noisename in noises:
                if noisename == 'Identity':
                    distortion = Identity()
                if noisename == 'Brightness':
                    distortion = Brightness()
                elif noisename == 'Contrast':
                    distortion = Contrast()
                elif noisename == 'Saturation':
                    distortion = Saturation()
                elif noisename == 'Hue':
                    distortion = Hue()
                elif noisename == 'Gaussian Noise':
                    distortion = GaussianNoise()
                elif noisename == 'Gaussian Blur':
                    distortion = GaussianBlur()
                elif noisename == 'Jpeg':
                    distortion = Jpeg()
                elif noisename == 'Resize':
                    distortion = Resize()
                elif noisename == 'Crop':
                    distortion = Crop()
                elif noisename == 'Rotate':
                    distortion = Rotate()
                elif noisename == 'Flip':
                    distortion = Flip()
                elif noisename == 'Perspective Transformation':
                    distortion = RandomCompensateTransformer()
                save_path_root = os.path.join(output_dir, noisename)
                if not os.path.exists(save_path_root):
                    os.makedirs(save_path_root)
                
                for f in noises[noisename]:
                    print("noise: {}, factor: {}".format(noisename, f))
                    save_path = os.path.join(save_path_root, '{}_{}_{}'.format(sele_size, iter, f))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    else:
                        for filename in os.listdir(save_path):
                            file_path = os.path.join(save_path, filename)
                            os.remove(file_path)
                    
                    for i in tqdm(range(100)):
                        awti_name = "{:03d}_awti.png".format(i)
                        awti_path = os.path.join(input_dir, awti_name)
                        awti = cv2.imread(awti_path, cv2.IMREAD_COLOR)
                        ti_name = "{:03d}_ti.png".format(i)
                        ti_path = os.path.join(input_dir, ti_name)
                        ti = cv2.imread(ti_path, cv2.IMREAD_COLOR)
                        
                        # brightness = Brightness()
                        # aw_image = brightness.test(aw_image, factor=1.3)
                        # cv2.imwrite('eval_watermark/img/aw_image_bright.png', aw_image)
                        # contrast = Contrast()
                        # aw_image = contrast.test(aw_image, factor=0.5)
                        # cv2.imwrite('eval_watermark/img/aw_image_contrast.png', aw_image)
                        # gaussianNoise = GaussianNoise()
                        # aw_image = gaussianNoise.test(aw_image, std=0.3)
                        # cv2.imwrite('eval_watermark/img/aw_image_noise.png', aw_image)
                        # gaussianBlur = GaussianBlur()
                        # aw_image = gaussianBlur.test(aw_image, sigma=1.2)
                        # cv2.imwrite('eval_watermark/img/aw_image_blur.png', aw_image)
                        
                        # resize = Resize()
                        # aw_image = resize.test(aw_image, scale_p=1.5)
                        # cv2.imwrite('eval_watermark/img/aw_image_resize.png', aw_image)
                        # aw_image = aw_image[20:, :, :]
                        # crop = Crop()
                        # aw_image = crop.test(aw_image, scale_p=0.5)
                        # cv2.imwrite('eval_watermark/img/aw_image_crop.png', aw_image)
                        # perspective = RandomCompensateTransformer()
                        # aw_image = perspective.test(aw_image, shift_d=2)
                        # cv2.imwrite('eval_watermark/img/aw_image_perspective.png', aw_image)

                        noised_awti = distortion.test(awti, ti, f)
                        noised_awti_path = os.path.join(save_path, awti_name)
                        cv2.imwrite(noised_awti_path, noised_awti)
