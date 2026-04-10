import sys

from natsort import natsorted
sys.path.append('/media/dongli911/Documents/Workflow/YanFangjun/UAW')
from Tools.ImageIO import *
from embed_utils import *
import time
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # patch settings
    parser.add_argument('--s', type=int, default=30, help='Patch size / scaling factor')
    parser.add_argument('--use_wm', type=bool, default=True, help='Use adversarial watermark (two patches)')
    parser.add_argument('--patch_dir', type=str, default='results/AllData_results/results_uawup_eps100/size=30_step=3_eps=100_lambdaw=0.01', help='UAW补丁目录')
    parser.add_argument('--root_eval', type=str, default='results/AllData_results/results_udup/size=30_step=3_eps=120_lambdaw=0.1', help='UA补丁目录')
    parser.add_argument('--patch_iter', type=int, default=27, help='使用的补丁迭代次数')
    parser.add_argument('--wm_h', type=int, default=8, help='Watermark height')
    parser.add_argument('--wm_w', type=int, default=8, help='Watermark width')

    # image settings
    parser.add_argument('--num', type=int, default=100, help='Number of images to process')
    parser.add_argument('--H', type=int, default=None, help='Text image height')
    parser.add_argument('--W', type=int, default=None, help='Text image width')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA for computation')
    parser.add_argument('--num_repeats', type=int, default=5, help='Number of repeats for underpainting tiling')
    parser.add_argument('--source_dir', type=str, default='AllData/web_page', help='Directory of input text images')
    parser.add_argument('--save_dir', type=str, default='results/eval_watermark/real/web_page', help='Directory to save watermarked images')

    args = parser.parse_args()

    
    wm_h, wm_w = args.wm_h, args.wm_w
    H, W = args.H, args.W
    use_cuda = args.use_cuda
    num_repeats = args.num_repeats
    s = args.s
    use_wm = args.use_wm
    source_dir = args.source_dir
    save_dir = args.save_dir
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 选择一、随机生成指定大小的水印信息
    # np.random.seed(42)
    # watermark = np.random.randint(0, 2, wm_h * wm_w).tolist()
    # wm = np.array(watermark).reshape(wm_h, wm_w)

    # 选择二、自定义水印矩阵
    wm = np.array(
        [[0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0]]
    )
    assert wm.shape == (wm_h, wm_w)

    watermark = wm.flatten().tolist()
    print('[', end='')
    for i in range(wm_h):
        print('[', end='')
        for j in range(wm_w):
            if j == wm_w-1:
                print(wm[i][j], end='')
            else:
                print(wm[i][j], end=', ')
        if i==wm_h-1:
            print(']]')
        else:
            print('],')

    text_paths = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    text_paths = natsorted(text_paths)[:args.num]
    
    # Initialize timing statistics for Watermark Modulation
    total_times = []
    module1_times = []  # Bit-template Mapping
    module2_times = []  # Flip-based Unit Construction
    module3_times = []  # Tiling for Underpainting Generation
    module4_times = []  # Fusion with Text Image
    
    if use_wm:  # 使用对抗水印（两个patch）
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                os.remove(file_path)

        patch1_path = os.path.join(args.patch_dir, "advpatch", f"advpatch1_{args.patch_iter}.png")
        patch2_path = os.path.join(args.patch_dir, "advpatch", f"advpatch2_{args.patch_iter}.png")
        # Load patches as torch tensors
        patch1 = img_read(patch1_path).to(device)
        patch2 = img_read(patch2_path).to(device)
        
        # Module 1: Bit-template Mapping (using torch tensors)
        t1 = time.perf_counter()
        base_block = bit_template_mapping(wm, patch1, patch2, device)
        t2 = time.perf_counter()
        module1_times.append(t2 - t1)
        
        # Module 2: Flip-based Unit Construction (not needed separately for this approach)
        # Module 3: Tiling for Underpainting Generation
        au_image = tiling_for_underpainting_generation(base_block, num_repeats, device)
        t3 = time.perf_counter()
        module2_times.append(0)  # Module 2 is embedded in Module 3
        module3_times.append(t3 - t2)
        
    else:  # 仅使用对抗样本（一个patch）
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # else:
        #     for filename in os.listdir(save_dir):
        #         file_path = os.path.join(save_dir, filename)
        #         os.remove(file_path)

        patch_path = os.path.join(args.root_eval, "advtorch", "advpatch_27")
        adv_patch = torch.load(patch_path).cuda()
        au_image = repeat_4D(adv_patch, s*wm_h*2*num_repeats, s*wm_w*2*num_repeats)
    
    au_img = au_image[:, :, :H, :W]
    au_img = au_image.to(device)
    delete_scaled_files(save_dir)
    
    for text_path in tqdm(text_paths):
        img_start_time = time.perf_counter()
        source_path = os.path.join(source_dir, text_path)  # 文本图像路径
        text_img = img_read(source_path).to(device)  # Load text image as tensor
        
        # Module 4: Fusion with Text Image
        awti_img = fusion_with_text_image(text_img, au_img, threshold=216/255, device=device)
        save_path = os.path.join(save_dir, text_path.replace('.png', '_awti.png').replace('_ti_awti.png', '_awti.png'))  # Save result
        img_write(save_path, awti_img[:, :, :H, :W])
        img_end_time = time.perf_counter()
        module4_times.append(img_end_time - img_start_time)
        total_times.append(img_end_time - img_start_time + sum([module1_times[0], module2_times[0], module3_times[0]]) if module1_times else img_end_time - img_start_time)
    
    if total_times:  # Print timing statistics
        log_path = os.path.join(save_dir, 'log')
        file = open(log_path, 'a')
        logINFO("\n" + "="*60, f=file)
        logINFO("[Timing Statistics] Watermark Modulation", f=file)
        logINFO("="*60, f=file)
        logINFO(f"Average total time per image: {np.mean(total_times):.4f} seconds", f=file)
        if module1_times:
            logINFO(f"  - Bit-template Mapping: {module1_times[0]:.4f} seconds", f=file)
        if module2_times:
            logINFO(f"  - Flip-based Unit Construction: {module2_times[0]:.4f} seconds", f=file)
        if module3_times:
            logINFO(f"  - Tiling for Underpainting Generation: {module3_times[0]:.4f} seconds", f=file)
        if module4_times:
            logINFO(f"  - Fusion with Text Image (overlay): {np.mean(module4_times):.4f} seconds", f=file)
        logINFO(f"Total images processed: {len(total_times)}", f=file)
        logINFO("="*60 + "\n", f=file)
