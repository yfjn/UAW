from embed_utils import *
import torch

if __name__ == "__main__":
    wm_h, wm_w = 8, 8
    use_cuda = True
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 选择一、随机生成指定大小的水印信息
    np.random.seed(42)
    watermark = np.random.randint(0, 2, wm_h * wm_w).tolist()
    wm = np.array(watermark).reshape(wm_h, wm_w)

    # 选择二、自定义水印矩阵
    # wm = np.array(
    #     [[1, 0, 0, 1, 1, 1, 1, 1, 0],
    #     [1, 1, 1, 1, 1, 1, 0, 1, 0],
    #     [0, 0, 1, 0, 1, 0, 1, 1, 0],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 0],
    #     [1, 0, 0, 1, 1, 0, 1, 0, 1],
    #     [0, 0, 1, 1, 1, 0, 1, 0, 1],
    #     [0, 1, 0, 0, 1, 1, 0, 1, 0],
    #     [1, 1, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 1, 0, 0]]
    # )
    # assert wm.shape == (wm_h, wm_w)

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
    
    # 输入文件路径和输出路径
    num_repeats = 5
    root_eval = os.path.join('results', 'AllData_results', 'results_uawup_eps100')
    special_eval=[
        # "size=10_step=3_eps=100_lambdaw=10",
        # "size=20_step=3_eps=100_lambdaw=1",
        # "size=30_step=3_eps=100_lambdaw=0.01",
        # "size=40_step=3_eps=100_lambdaw=0.001",
        # "size=50_step=3_eps=100_lambdaw=0.01"
        "size=30_step=3_eps=100_lambdaw=0.01_lambday=1e-05",
    ]

    save_dir = f'results/eval_watermark/size_it_{wm_h}×{wm_w}_mse'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_awu = os.path.join(save_dir, 'awu')
    if not os.path.exists(save_awu):
        os.makedirs(save_awu)
    # else:
    #     for filename in os.listdir(save_dir):
    #         file_path = os.path.join(save_dir, filename)
    #         os.remove(file_path)

    # 生成并保存水印图像
    for dir in special_eval:
        sele_size=dir.split('=')[1].split('_')[0]
        s = int(sele_size)
        iter_eval = gen_iter_dict(root_eval,dir)
        iter_eval = sorted(iter_eval,key=lambda x:int(x))
        iter_eval = iter_eval[3:4]  # MUI=0.09
        # iter_eval = iter_eval[6:7]  # MUI=0.12

        for iter in iter_eval:
            print("dir:", dir, "iter:", iter)
            patch1_path = os.path.join(root_eval, dir, "advpatch", "advpatch1_{}.png".format(iter))
            patch2_path = os.path.join(root_eval, dir, "advpatch", "advpatch2_{}.png".format(iter))
            awu_path = "{}/awu_{}_{}.png".format(save_awu, sele_size, iter)  # 对抗水印底纹路径
            
            # 加载patch图像为tensor
            from Tools.ImageIO import img_read, img_write
            patch1 = img_read(patch1_path).to(device)
            patch2 = img_read(patch2_path).to(device)
            
            # Module 1: Bit-template Mapping
            base_block = bit_template_mapping(wm, patch1, patch2, device)
            
            # Module 3: Tiling for Underpainting Generation
            awu_image_t = tiling_for_underpainting_generation(base_block, num_repeats, device)
            img_write(awu_path, awu_image_t)

            output_dir = "{}/awti_{}_{}".format(save_dir, sele_size, iter)  # 对抗水印文本图像路径
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # else:
            #     for filename in os.listdir(output_dir):
            #         file_path = os.path.join(output_dir, filename)
            #         os.remove(file_path)
            
            for i in tqdm(range(100)):
                save_name = "{:03d}".format(i)
                save_path = os.path.join(output_dir, save_name)
                
                # 生成文本图像
                width = s * wm_w * num_repeats
                height = s * wm_h * num_repeats
                text_image_t = generate_text_image(width, height, s, font_size=None, font_color=None, device=device)
                img_write(save_path + '_ti.png', text_image_t)
                
                # Module 4: Fusion with Text Image
                awti_image_t = fusion_with_text_image(text_image_t, awu_image_t, threshold=216, device=device)
                img_write(save_path + '_awti.png', awti_image_t)
