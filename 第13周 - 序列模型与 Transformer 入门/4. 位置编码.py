# 注意力本身不知道词的顺序("我爱你"和"你爱我"对它来说一样),所以必须显式注入位置信息:
import math

import torch


def sinusoidal_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


pe = sinusoidal_positional_encoding(50, 64)
print(pe.shape)  # [50, 64]

import matplotlib.pyplot as plt

plt.imshow(pe, cmap="RdBu", aspect="auto")
plt.xlabel("维度")
plt.ylabel("位置")
plt.colorbar()
plt.title("正弦位置编码")
plt.show()
