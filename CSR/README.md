# Camera Shooting Resilient (CSR) Watermarking 实现说明

基于论文 "A Camera Shooting Resilient Watermarking Scheme for Underpainting Documents" (Fang et al., IEEE TCSVT 2020) 复现

## 算法原理

### 核心思想
该算法通过在文档底纹（underpainting）中嵌入水印来实现泄密追踪。相比传统水印方案，具有以下优势：

1. **隐蔽性（Inconspicuousness）**：水印嵌入在背景中，不影响正常阅读
2. **鲁棒性（Robustness）**：基于DCT系数交换，能抵抗相机拍摄带来的各种失真
3. **自相关性（Autocorrelation）**：采用翻转排列，即使只拍摄部分文档也能提取水印

### 嵌入流程

```
消息序列 → CRC编码 → BCH编码 → 二进制矩阵W
                                    ↓
原始底纹 → 颜色优化 → 高斯噪声 → DCT嵌入 → 缩放翻转 → 水印底纹
```

### 关键方程

**1. 底纹优化 (Eq. 1)**

\[
\min_{P'_o} \|P_o - P'_o\|_2^2 \quad \text{s.t.} \quad th_1 \leq Y(P'_o) \leq th_2
\]

**2. DCT系数嵌入 (Eq. 3)**

\[
\begin{cases}
C_1 = r, C_2 = -r & \text{if } w = 0 \\
C_1 = -r, C_2 = r & \text{otherwise}
\end{cases}
\]

**3. 翻转对称性 (Eq. 4)**

\[
P_f(x, y) = P_f(2N-x, y) = P_f(x, 2N-y) = P(x, y)
\]

### 提取流程

```
拍摄图像 → 透视校正 → 直方图均衡 → 对称性定位 → 文本补偿 → DCT提取 → BCH/CRC解码
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_strength` | 30.0 | DCT系数修改强度，越大越鲁棒但越可见 |
| `block_size` | 16 | 嵌入块大小（像素） |
| `watermark_rows` | 16 | 水印矩阵行数 \(a\) |
| `watermark_cols` | 8 | 水印矩阵列数 \(b\) |
| `underpainting_color` | (199, 237, 204) | RGB底纹颜色 |

## 文件说明

- `csr_watermark.py` - 主要实现代码

## 算法限制

1. **消息容量**：48比特（约6个字符），使用BCH(127,64)纠错码
2. **拍摄角度**：左右60°以内，上下60°以内
3. **拍摄距离**：15cm-85cm
4. **纠错能力**：最多纠正10比特错误

## 改进建议

实际应用中可考虑以下改进：

1. 使用更强的纠错码（如LDPC）提高纠错能力
2. 实现自适应嵌入强度
3. 添加深度学习辅助的水印定位
4. 支持彩色底纹水印

## 参考文献

```
@article{fang2020camera,
  title={A Camera Shooting Resilient Watermarking Scheme for Underpainting Documents},
  author={Fang, Han and Zhang, Weiming and Ma, Zehua and Zhou, Hang and Sun, Shan and Cui, Hao and Yu, Nenghai},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={30},
  number={11},
  pages={4075--4089},
  year={2020}
}
```
