from embed_utils import *
import torch
from Tools.ImageIO import img_read, img_write

if __name__ == "__main__":
    # 固定随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
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
    save_dir = 'results/eval_watermark/size_it/awu/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, 'random')
    
    # 生成并保存水印图像
    s = 30
    num_repeats = 5
    value_range = (155, 256)  # 像素值范围为[low, high)
    patch1_np = np.random.randint(*value_range, (s,s,3), dtype=np.uint8)  # 均匀分布 (Uniform Distribution)
    patch2_np = np.random.randint(*value_range, (s,s,3), dtype=np.uint8)  # 高斯分布 (正态分布)：np.random.normal(loc=均值, scale=标准差, size=形状)
    patch1_path = save_dir + "_patch1.png"
    patch2_path = save_dir + "_patch2.png"
    cv2.imwrite(patch1_path, patch1_np)  # 保存patch1
    cv2.imwrite(patch2_path, patch2_np)  # 保存patch2
    
    # 转换为tensor
    patch1 = torch.from_numpy(patch1_np).float().to(device)
    patch2 = torch.from_numpy(patch2_np).float().to(device)

    awu_path = save_dir + "_awu.png"  # 对抗水印底纹路径
    
    # Module 1: Bit-template Mapping
    base_block = bit_template_mapping(wm, patch1, patch2, device)
    
    # Module 3: Tiling for Underpainting Generation
    awu_image_t = tiling_for_underpainting_generation(base_block, num_repeats, device)
    img_write(awu_path, awu_image_t)
    
    output_dir = "eval_watermark/size_it/awti_random"  # 对抗水印文本图像路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            os.remove(file_path)
    
    for i in tqdm(range(100)):
        save_name = "{:03d}".format(i+1)
        save_path = os.path.join(output_dir, save_name)
        
        # 生成文本图像
        width = s * 8 * num_repeats
        height = s * 8 * num_repeats
        text_image_t = generate_text_image(width, height, s, font_size=None, font_color=None, device=device)
        img_write(save_path + '_ti.png', text_image_t)
        
        # Module 4: Fusion with Text Image
        awti_image_t = fusion_with_text_image(text_image_t, awu_image_t, threshold=216, device=device)
        img_write(save_path + '_awti.png', awti_image_t)
