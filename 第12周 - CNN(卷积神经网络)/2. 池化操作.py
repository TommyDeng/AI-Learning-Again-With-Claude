import torch
from torch import nn

x = (
    torch.tensor(
        [
            [1.0, 3.0, 2.0, 4.0],
            [5.0, 6.0, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 1.0, 2.0, 0.0],
        ]
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

maxpool = nn.MaxPool2d(kernel_size=2)
output = maxpool(x)
print("原图:\n", x.squeeze())
print("\n最大池化后(尺寸减半):\n", output.squeeze())

# 为什么需要池化:
# 池化(通常是取窗口内最大值)会缩小特征图的尺寸,
# 一方面减少计算量, 另一方面让网络对"位置的微小偏移"更鲁棒
# 比如一张猫的图片，猫稍微往左挪了几个像素，池化后的特征基本不变。
