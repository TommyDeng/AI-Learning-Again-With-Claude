import math

import torch
import torch.nn.functional as F
from torch import nn


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # 每个词和每个词的相似度
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)  # 变成概率分布(每行和为1)
    return weights @ V, weights  # 按权重对V加权求和


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)  # 一次性算出Q、K、V
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # 各自 [B, heads, T, d_k]
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # 拼接所有头
        return self.out(out)


mha = MultiHeadAttention(d_model=64, n_heads=4)
print(mha(torch.randn(2, 10, 64)).shape)  # [2, 10, 64]

# 为什么要"多头": 一个头只能学一种"关注模式"。多个头并行,可以分别关注语法关系、指代关系、位置邻近等不同方面,最后拼接起来。
