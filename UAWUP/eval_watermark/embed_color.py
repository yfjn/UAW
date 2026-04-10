from embed_utils import *
import torch
from Tools.ImageIO import img_read, img_write

if __name__ == "__main__":
    use_cuda = True
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 随机生成64位水印信息
    # watermark = np.random.randint(0, 2, 64).tolist()
    # wm = np.array(watermark).reshape(8, 8)
    wm = np.array(
        [[1, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 1]]
    )
    watermark = wm.flatten().tolist()
    print('[', end='')
    for i in range(8):
        print('[', end='')
        for j in range(8):
            if j == 7:
                print(wm[i][j], end='')
            else:
                print(wm[i][j], end=', ')
        if i==7:
            print(']]')
        else:
            print('],')
    
    # 输入文件路径和输出路径
    num_repeats = 5
    s = 30
    root_eval = os.path.join('results', 'AllData_results', 'results_uawup_eps100')
    color_eval=[
        'black',
        'blue',
        'gray',
        'green',
        'red',
        'yellow',
    ]
    
    save_dir = 'eval_watermark/color/awu'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # else:
    #     for filename in os.listdir(save_dir):
    #         file_path = os.path.join(save_dir, filename)
    #         os.remove(file_path)

    # 生成并保存水印图像
    patch1_path = os.path.join(root_eval, "size=30_step=3_eps=100_lambdaw=0.01", "advpatch", "advpatch1_27.png")
    patch2_path = os.path.join(root_eval, "size=30_step=3_eps=100_lambdaw=0.01", "advpatch", "advpatch2_27.png")
    awu_path = "eval_watermark/color/awu/awu.png"  # 对抗水印底纹路径
    
    # 加载patch图像为tensor
    patch1 = img_read(patch1_path).to(device)
    patch2 = img_read(patch2_path).to(device)
    
    # Module 1: Bit-template Mapping
    base_block = bit_template_mapping(wm, patch1, patch2, device)
    
    # Module 3: Tiling for Underpainting Generation
    awu_image_t = tiling_for_underpainting_generation(base_block, num_repeats, device)
    img_write(awu_path, awu_image_t)

    for color in color_eval:
        print("color:", color)
        output_dir = "eval_watermark/color/awti_{}".format(color)  # 对抗水印文本图像路径
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                os.remove(file_path)
        
        for i in tqdm(range(100)):
            save_name = "{:03d}".format(i)
            save_path = os.path.join(output_dir, save_name)
            
            # 生成文本图像
            width = s * 8 * num_repeats
            height = s * 8 * num_repeats
            text_image_t = generate_text_image(width, height, s, font_size=None, font_color=color, device=device)
            img_write(save_path + '_ti.png', text_image_t)
            
            # Module 4: Fusion with Text Image
            awti_image_t = fusion_with_text_image(text_image_t, awu_image_t, threshold=216, device=device)
            img_write(save_path + '_awti.png', awti_image_t)
