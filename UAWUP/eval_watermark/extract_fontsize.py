from extract_utils import *
import torch
import sys
sys.path.append('/media/dongli911/Documents/Workflow/YanFangjun/UAW')
from Tools.ImageIO import img_read, img_write

if __name__ == "__main__":
    use_cuda = True
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 定义真实的8x8水印矩阵
    true_watermark = np.array(
        [[1, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 1]]
    )

    # 输入文件路径和输出路径
    patch_dir = os.path.join('results', 'AllData_results', 'results_uawup_eps100', 'size=30_step=3_eps=100_lambdaw=0.01')
    p_0_path = os.path.join(patch_dir, "advpatch", "advpatch1_27.png")
    p_1_path = os.path.join(patch_dir, "advpatch", "advpatch2_27.png")
    sele_size = patch_dir.split('/')[-1].split('=')[1].split('_')[0]
    s = int(sele_size)

    num_repeats = 8
    fontsize_eval = [
        15,
        30,
        45,
    ]

    for fontsize in fontsize_eval:
        print("fontsize:", fontsize)
        awti_dir = 'eval_watermark/fontsize/awti_{}'.format(fontsize)
        delete_scaled_files(awti_dir)
        
        # 生成并保存水印图像
        log_path = os.path.join(awti_dir, "log.txt")
        log_file = open(log_path, 'w')
        
        acc_before_recovers = []
        acc_after_recovers = []
        for i in tqdm(range(100)):
            save_name = "{:03d}".format(i)
            logINFO(save_name, log_file, file=True)
            save_path = os.path.join(awti_dir, save_name)

            # Load watermarked image and templates as tensors [C,H,W] RGB [0,1]
            awti_img = img_read(save_path + '_awti.png').squeeze(0).to(device)
            p_0 = img_read(p_0_path).squeeze(0).to(device)
            p_1 = img_read(p_1_path).squeeze(0).to(device)
            
            # Module 1: Watermark Block Synchronization
            xy0, XY0 = watermark_block_synchronization(
                awti_img, s, num_repeats, wm_h=8, wm_w=8, 
                device=device, save_path=save_path, f=log_file
            )
            
            # Module 2: Watermark Block State Determination
            awu_image_t = watermark_block_state_determination(
                awti_img, s, xy0, XY0, wm_h=8, wm_w=8, 
                device=device, save_path=save_path, f=log_file
            )
            
            # Module 3: Watermark Message Extraction
            acc_before_recover = watermark_message_extraction(
                awti_img, p_0, p_1, s, xy0, XY0, true_watermark, 
                wm_h=8, wm_w=8, device=device, f=log_file
            )
            acc_after_recover = watermark_message_extraction(
                awu_image_t, p_0, p_1, s, xy0, XY0, true_watermark, 
                wm_h=8, wm_w=8, device=device, f=log_file
            )

            acc_before_recovers.append(acc_before_recover)
            acc_after_recovers.append(acc_after_recover)
            logINFO('', log_file)

        logINFO(f"{fontsize} 平均恢复前比特正确率: {np.mean(acc_before_recovers) * 100:.2f}%", log_file, file=True, console=True)
        logINFO(f"{fontsize} 平均恢复后比特正确率: {np.mean(acc_after_recovers) * 100:.2f}%", log_file, file=True, console=True)
