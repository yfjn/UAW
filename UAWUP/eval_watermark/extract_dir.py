import sys
from extract_utils import *
from tqdm import tqdm
import time
import torch
import argparse
from natsort import natsorted
import sys
sys.path.append('/media/dongli911/Documents/Workflow/YanFangjun/UAW')
from Tools.ImageIO import img_read, img_write

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=bool, default=False, help='Use CUDA for computation')
    parser.add_argument('--s', type=int, default=30, help='Patch size / scaling factor')
    parser.add_argument('--patch_dir', type=str, default='results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01', help='UAW补丁目录')
    parser.add_argument('--patch_iter', type=int, default=27, help='使用的补丁迭代次数')

    parser.add_argument('--awti_dir', type=str, default='results/eval_watermark/real/web_page', help='输入图像目录')
    parser.add_argument('--res', type=int, default=960, help='Image resolution')
    args = parser.parse_args()
    
    use_cuda = args.use_cuda
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

    patch_dir = args.patch_dir
    s = args.s
    # 针对real
    p_0_path = os.path.join(patch_dir, "advpatch", f"advpatch1_{args.patch_iter}.png")
    p_1_path = os.path.join(patch_dir, "advpatch", f"advpatch2_{args.patch_iter}.png")
    
    # Load patches as torch tensors
    p_0_tensor = img_read(p_0_path).to(device)
    p_1_tensor = img_read(p_1_path).to(device)
    num_repeats = 0  # (col_repeats, row_repeats)

    # 针对random
    # p_0_path = os.path.join(patch_dir, 'random_patch1.png')
    # p_1_path = os.path.join(patch_dir, 'random_patch2.png')
    # test_paths = [f.replace('_awti.png', '') for f in os.listdir(awti_dir) if f.endswith('_awti.png')]
    # num_repeats = 10

    
    # 生成并保存水印图像
    awti_dir = args.awti_dir
    delete_scaled_files(awti_dir)
    # exit(0)
    test_paths = natsorted([f for f in os.listdir(awti_dir) if os.path.splitext(f)[1].lower() in Image.registered_extensions()])
    log_path = os.path.join(awti_dir, "log.txt")
    log_file = open(log_path, 'w')
    
    # Initialize timing statistics for Watermark Extraction
    total_times = []
    module1_times = []  # Watermark Block Synchronization
    module2_times = []  # Watermark Block State Determination
    module3_times = []  # Watermark Message Extraction
    
    acc_before_recovers = []
    acc_after_recovers = []
    for test_path in tqdm(test_paths):
        img_start_time = time.perf_counter()
        img_path = os.path.join(awti_dir, test_path)
        img_name = os.path.splitext(test_path)[0].replace('_awti', '')
        save_path = os.path.join(awti_dir, img_name)  # 对抗水印文本图像路径
        logINFO(save_path, log_file)
        # Load watermarked image and templates as tensors [C,H,W] RGB [0,1]
        awti_img = img_read(img_path).squeeze(0).to(device)
        awti_img = awti_img[:, :args.res, :args.res]  # Crop to specified resolution
        p_0 = img_read(p_0_path).squeeze(0).to(device)
        p_1 = img_read(p_1_path).squeeze(0).to(device)
        
        # Module 1: Watermark Block Synchronization
        t1 = time.perf_counter()
        xy0, XY0 = watermark_block_synchronization(awti_img, s, num_repeats, wm_h=8, wm_w=8, device=device, save_path=save_path, f=log_file)
        t2 = time.perf_counter()
        module1_times.append(t2 - t1)
        
        # Module 2: Watermark Block State Determination
        awu_image_t = watermark_block_state_determination(awti_img, s, xy0, XY0, wm_h=8, wm_w=8, device=device, save_path=save_path, f=log_file)
        t3 = time.perf_counter()
        module2_times.append(t3 - t2)
        
        # Module 3: Watermark Message Extraction
        # Extract from cropped awti (before recover)
        acc_before_recover = watermark_message_extraction(awti_img, p_0, p_1, s, xy0, XY0, true_watermark, wm_h=8, wm_w=8, device=device, f=log_file)
        t4 = time.perf_counter()
        # Extract from recovered awu (after recover) - pass tensor directly
        acc_after_recover = watermark_message_extraction(awu_image_t, p_0, p_1, s, xy0, XY0, true_watermark, wm_h=8, wm_w=8, device=device, f=log_file)
        t5 = time.perf_counter()
        module3_times.append(t5 - t4)
        
        img_end_time = time.perf_counter()
        total_times.append(img_end_time - img_start_time)
        
        # Log to file
        logINFO(f"Acc before: {acc_before_recover:.4f}, Acc after: {acc_after_recover:.4f}", log_file)

        acc_before_recovers.append(acc_before_recover)
        acc_after_recovers.append(acc_after_recover)
        logINFO('', log_file)

    # Print timing statistics
    if total_times:
        logINFO("\n" + "="*60, log_file)
        logINFO("[Timing Statistics] Watermark Extraction", log_file)
        logINFO("="*60, log_file)
        logINFO(f"Average total time per image: {np.mean(total_times):.4f} seconds", log_file)
        if module1_times:
            logINFO(f"  - Watermark Block Synchronization: {np.mean(module1_times):.4f} seconds", log_file)
        if module2_times:
            logINFO(f"  - Watermark Block State Determination: {np.mean(module2_times):.4f} seconds", log_file)
        if module3_times:
            logINFO(f"  - Watermark Message Extraction: {np.mean(module3_times):.4f} seconds", log_file)
        logINFO(f"Total images processed: {len(total_times)}", log_file)
        logINFO("="*60 + "\n", log_file)

    logINFO(f"平均恢复前比特正确率: {np.mean(acc_before_recovers) * 100:.2f}%", log_file)
    logINFO(f"平均恢复后比特正确率: {np.mean(acc_after_recovers) * 100:.2f}%", log_file)
