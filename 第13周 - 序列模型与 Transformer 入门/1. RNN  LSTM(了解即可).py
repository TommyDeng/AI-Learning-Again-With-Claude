import torch
from torch import nn

# LSTM输入形状: [batch, 序列长度, 每步特征数]
lstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=True)

x = torch.randn(4, 10, 8)  # 4个样本,每个10步,每步8维特征
output, (h_n, c_n) = lstm(x)

print("output:", output.shape)  # [4, 10, 16] 每一步的输出
print("h_n:", h_n.shape)  # [1, 4, 16]  最后一步的隐藏状态
print("c_n:", c_n.shape)  # [1, 4, 16]  最后一步的细胞状态


# 为什么RNN/LSTM被取代:

# 它必须一步一步按顺序算,无法并行,训练很慢
# 序列很长时,早期信息容易在传递中"遗忘"(长距离依赖问题)

# Transformer 用注意力机制一次性看到整个序列,同时解决了这两个问题。
