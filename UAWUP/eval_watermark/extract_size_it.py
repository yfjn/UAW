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
    
    # 定义真实的水印矩阵
    true_watermark = np.array(
[[1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
[0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
[1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
[0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    )
    wm_h, wm_w = true_watermark.shape
    # assert true_watermark.shape == (wm_h, wm_w)
    
    # 输入文件路径和输出路径
    
    root_eval = os.path.join('results', 'AllData_results', 'results_uawup_eps100')
    special_eval=[
        # "size=10_step=3_eps=100_lambdaw=10",
        # "size=20_step=3_eps=100_lambdaw=1",
        # "size=30_step=3_eps=100_lambdaw=0.01",
        # "size=40_step=3_eps=100_lambdaw=0.001",
        # "size=50_step=3_eps=100_lambdaw=0.01",
        "size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05",
    ]
    num_repeats = 6
    save_dir = f'results/eval_watermark/size_it_{wm_h}×{wm_w}_mse'
    log_dir = os.path.join(save_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成并保存水印图像
    # The error occurs because OpenCV is trying to use its bundled Qt libraries,
    # but can't properly initialize the XCB platform plugin which is needed for GUI functionality on X11-based Linux systems.
    # Set QT_QPA_PLATFORM environment variable: export QT_QPA_PLATFORM=offscreen, python eval_watermark/extract_batch.py
    for dir in special_eval:
        sele_size = dir.split('=')[1].split('_')[0]
        s = int(sele_size)
        iter_eval = gen_iter_dict(root_eval,dir)
        iter_eval = sorted(iter_eval,key=lambda x:int(x))
        iter_eval = iter_eval[3:4]  # MUI=0.09

        for iter in iter_eval:
            print("dir:", dir, "iter:", iter)
            p_0_path = os.path.join(root_eval, dir, "advpatch", "advpatch1_{}.png".format(iter))
            p_1_path = os.path.join(root_eval, dir, "advpatch", "advpatch2_{}.png".format(iter))
            awti_dir_path = "{}/awti_{}_{}".format(save_dir, sele_size, iter)
            delete_scaled_files(awti_dir_path)
            log_path = os.path.join(log_dir, "log_{}_{}.txt".format(sele_size, iter))
            log_file = open(log_path, 'a')
            
            acc_before_recovers = []
            acc_after_recovers = []
            for i in tqdm(range(100)):
                save_name = "{:03d}".format(i)
                logINFO(save_name, log_file)
                save_path = os.path.join(awti_dir_path, save_name)
                
                # Load watermarked image and templates as tensors [C,H,W] RGB [0,1]
                awti_img = img_read(save_path + '_awti.png').squeeze(0).to(device)
                p_0 = img_read(p_0_path).squeeze(0).to(device)
                p_1 = img_read(p_1_path).squeeze(0).to(device)
                
                # Module 1: Watermark Block Synchronization
                col_peaks, row_peaks, wh, xy0, XY0 = watermark_block_synchronization(
                    awti_img, s, num_repeats, wm_h=wm_h, wm_w=wm_w, device=device, save_path=save_path, f=log_file
                )
                
                # Module 2: Watermark Block State Determination
                awu_image_t = watermark_block_state_determination(
                    awti_img, s, xy0, XY0, wm_h=wm_h, wm_w=wm_w, device=device, save_path=save_path, f=log_file
                )
                
                # Module 3: Watermark Message Extraction
                acc_before_recover = watermark_message_extraction(
                    awti_img, p_0, p_1, s, xy0, XY0, true_watermark, wm_h=wm_h, wm_w=wm_w, device=device, save_path=save_path, f=log_file
                )
                acc_after_recover = watermark_message_extraction(
                    awu_image_t, p_0, p_1, s, xy0, XY0, true_watermark, wm_h=wm_h, wm_w=wm_w, device=device, save_path=save_path, f=log_file
                )

                acc_before_recovers.append(acc_before_recover)
                acc_after_recovers.append(acc_after_recover)
                logINFO('', log_file)

            logINFO(f"{sele_size} {iter} 平均恢复前比特正确率: {np.mean(acc_before_recovers) * 100:.2f}%", log_file, console=True)
            logINFO(f"{sele_size} {iter} 平均恢复后比特正确率: {np.mean(acc_after_recovers) * 100:.2f}%", log_file, console=True)
