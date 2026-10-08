import torch
from torch import nn

# 手动看一次卷积在做什么:用一个简单的边缘检测卷积核
image = (
    torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
        ]
    )
    .unsqueeze(0)
    .unsqueeze(0)
)  # 变成 [batch, channel, H, W] 的形状

# 一个检测"垂直边缘"的卷积核
edge_kernel = (
    torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    .unsqueeze(0)
    .unsqueeze(0)
)

# 直觉理解:
#       卷积核就是一个小窗口，在图像上滑动，每个位置做"逐元素相乘再求和"。
#       不同的卷积核能检测不同的模式(边缘、纹理、颜色块等)。
#
#       传统图像处理里这些核是人工设计的，CNN最厉害的地方是:这些核的数值是通过反向传播自动学出来的，不需要人工设计。

conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)
conv.weight.data = edge_kernel

output = conv(image)
print("原图:\n", image.squeeze())
print("\n卷积后(边缘被检测出来):\n", output.squeeze())
