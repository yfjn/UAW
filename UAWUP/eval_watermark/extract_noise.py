from extract_utils import *
import torch
import sys
sys.path.append('/media/dongli911/Documents/Workflow/YanFangjun/UAW')
from Tools.ImageIO import img_read, img_write

if __name__ == "__main__":
    # Device configuration
    use_cuda = True
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 定义真实的8x8水印矩阵
    true_watermark = np.array(
        [[0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0]]
    )

    # 输入文件路径和输出路径
    
    patch_dir = os.path.join('results', 'AllData_results', 'results_uawup_eps100', 'size=30_step=3_eps=100_lambdaw=0.01')
    p_0_path = os.path.join(patch_dir, "advpatch", "advpatch1_27.png")
    p_1_path = os.path.join(patch_dir, "advpatch", "advpatch2_27.png")
    sele_size = patch_dir.split('/')[-1].split('=')[1].split('_')[0]
    s = int(sele_size)
    num_repeats = 8

    root_eval = 'eval_watermark/noise'
    noises={
        "Identity": [1.],
        "Brightness": [0.7, 0.85, 1.15, 1.3],
        "Contrast": [0.7, 0.85, 1.15, 1.3],
        "Saturation": [0.7, 0.85, 1.15, 1.3],
        "Hue": [-0.2, -0.1, 0.1, 0.2],
        "Gaussian Noise": [0.05, 0.1, 0.15, 0.2],
        "Gaussian Blur": [1.5, 2.0],  # 0.5, 1.0, 
        "Jpeg": [90, 70, 50, 30, 10],
        "Resize": [0.6, 0.8, 1.2, 1.5],
        "Crop": [0.6, 0.8],
        "Rotate": [90, 180, 270],
        "Flip": [0, 1, -1],
    }
    
    # 生成并保存水印图像
    # for noise_dir in os.listdir(root_eval):
    # for noise_dir in noises:
    #     for f_dir in os.listdir(os.path.join(root_eval, noise_dir)):  # [1:]
            # if noise_dir == 'Crop':
            #     if f_dir.split('_')[-1] == '0.8':
            #         num_repeats = 6
            #     elif f_dir.split('_')[-1] == '0.6':
            #         num_repeats = 5
            # else:
            #     num_repeats = 8
    for noise_dir in noises:
        for f_dir in noises[noise_dir]:
            print(noise_dir, f_dir)  # , num_repeats
            f_dir = '{}_{}_{}'.format(30, 27, f_dir)
            delete_scaled_files(os.path.join(root_eval, noise_dir, f_dir))
            log_path = os.path.join(root_eval, noise_dir, f_dir, "log.txt")
            log_file = open(log_path, 'w')
            
            acc_before_recovers = []
            acc_after_recovers = []
            for i in tqdm(range(100)):
                save_name = "{:03d}".format(i)
                logINFO(save_name, log_file, file=True)
                save_path = os.path.join(root_eval, noise_dir, f_dir, save_name)

                # Load watermarked image and templates as tensors [C,H,W] RGB [0,1]
                awti_img = img_read(save_path + '_awti.png').squeeze(0).to(device)
                p_0 = img_read(p_0_path).squeeze(0).to(device)
                p_1 = img_read(p_1_path).squeeze(0).to(device)
                
                # Module 1: Watermark Block Synchronization
                col_peaks, row_peaks, wh, xy0, XY0 = watermark_block_synchronization(
                    awti_img, s, num_repeats, wm_h=8, wm_w=8, device=device, save_path=save_path, f=log_file
                )
                
                # Module 2: Watermark Block State Determination
                awu_image_t = watermark_block_state_determination(
                    awti_img, s, xy0, XY0, wm_h=8, wm_w=8, device=device, save_path=save_path, f=log_file
                )
                
                # Module 3: Watermark Message Extraction
                acc_before_recover = watermark_message_extraction(
                    awti_img, p_0, p_1, s, xy0, XY0, true_watermark, wm_h=8, wm_w=8, device=device, save_path=save_path, f=log_file
                )
                acc_after_recover = watermark_message_extraction(
                    awu_image_t, p_0, p_1, s, xy0, XY0, true_watermark, wm_h=8, wm_w=8, device=device, save_path=save_path, f=log_file
                )

                acc_before_recovers.append(acc_before_recover)
                acc_after_recovers.append(acc_after_recover)
                logINFO('', log_file)

            logINFO(f"{noise_dir} {f_dir} 平均恢复前比特正确率: {np.mean(acc_before_recovers) * 100:.2f}%", log_file, file=True, console=True)
            logINFO(f"{noise_dir} {f_dir} 平均恢复后比特正确率: {np.mean(acc_after_recovers) * 100:.2f}%", log_file, file=True, console=True)
