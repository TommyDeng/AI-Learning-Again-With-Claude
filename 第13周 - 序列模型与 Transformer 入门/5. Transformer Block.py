# Transformer Block(Transformer 模块) - 残差连接（Residual Connection）与层归一化（LayerNorm）
# Transformer 模块其内部结构包含两个主要部分
# 1. 多头自注意力机制MHA (self.attn)
# 2. 前馈神经网络 (self.ff)
import math

import torch.nn.functional as F
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)  # 残差连接 + 注意力
        x = x + self.ff(self.ln2(x))  # 残差连接 + 前馈网络
        return x


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


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # 每个词和每个词的相似度
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)  # 变成概率分布(每行和为1)
    return weights @ V, weights  # 按权重对V加权求和


# 两个关键点:

# 残差连接(x + ...) — 就是第12周提到的 ResNet 思想,让深层网络能顺利训练
# LayerNorm — 和第11周的 BatchNorm 作用类似,但是沿特征维度归一化,更适合序列数据
