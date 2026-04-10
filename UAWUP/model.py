import torch
import torch.nn as nn
import torchvision.models as models

class WatermarkExtractor(nn.Module):
    """
    基于 ConvNeXt-small 的水印提取器
    输入：尺寸为 (batch_size, 3, H, W) 的水印块图像
    输出：尺寸为 (batch_size, bit_num) 的比特串（每个比特在 [0,1] 范围内）
    
    参数：
        bit_num (int): 待提取比特的个数，默认64
    """
    def __init__(self, bit_num=64):
        super(WatermarkExtractor, self).__init__()
        self.bit_num = bit_num
        # 加载 ConvNeXt-small 模型，不使用预训练权重
        self.backbone = models.convnext_small(weights=None)
        # 修改分类器部分：原始 classifier 为 nn.Sequential(LayerNorm, Linear)
        # 我们将 Linear 层替换为输出 bit_num，并在后面接上 Sigmoid
        self.backbone.classifier = nn.Sequential(
            self.backbone.classifier[0],  # LayerNorm层
            self.backbone.classifier[1],  # Flatten层
            nn.Linear(768, bit_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播
        """
        return self.backbone(x)


class SimpleWatermarkExtractor(nn.Module):
    """
    简单的 CNN 水印提取器
    输入：尺寸为 (batch_size, 3, 240, 240) 的图像
    输出：尺寸为 (batch_size, bit_num) 的比特向量（经过 Sigmoid 激活）
    
    参数：
        bit_num (int): 待提取的比特个数，默认64
    """
    def __init__(self, bit_num=64):
        super(SimpleWatermarkExtractor, self).__init__()

        # 卷积层特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 输出尺寸: [B, 32, 120, 120]

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 输出尺寸: [B, 64, 60, 60]

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 输出尺寸: [B, 128, 30, 30]

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 输出尺寸: [B, 256, 15, 15]
        )
        
        # 全连接层分类/回归部分
        # 最终输出 bit_num 维度（对应需要提取的比特数）
        self.classifier = nn.Sequential(
            nn.Linear(256 * 15 * 15, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.),
            nn.Linear(1024, bit_num),
            nn.Sigmoid()  # 输出 [B, bit_num]
        )

    def forward(self, x):
        # x: [B, 3, 240, 240]
        x = self.features(x)
        # 展平为 [B, 特征数]
        x = x.view(x.size(0), -1)
        # 全连接层得到 logits
        x = self.classifier(x)
        return x


# 示例：创建模型并打印结构
if __name__ == '__main__':
    # model = WatermarkExtractor(bit_num=64)
    model = SimpleWatermarkExtractor(bit_num=64)
    print(model)
    # 输入假数据 (batch_size, 3, 240, 240)
    x = torch.randn(1, 3, 240, 240)
    bits = model(x)
    print("提取比特形状:", bits.shape)
