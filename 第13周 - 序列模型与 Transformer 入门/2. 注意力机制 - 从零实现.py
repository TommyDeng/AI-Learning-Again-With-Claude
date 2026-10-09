# 核心公式:Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # 每个词和每个词的相似度
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)  # 变成概率分布(每行和为1)
    return weights @ V, weights  # 按权重对V加权求和


# 模拟一句话:4个词,每个词用8维向量表示
torch.manual_seed(0)
x = torch.randn(1, 4, 8)

out, weights = scaled_dot_product_attention(x, x, x)  # Q=K=V=x,就是"自注意力"
print("注意力权重(每行是一个词对所有词的关注度):\n", weights[0])
print("输出形状:", out.shape)  # [1, 4, 8],形状不变,但每个词已融合了上下文

# 直觉理解: 每个词发出一个"查询(Q)",去和所有词的"键(K)"比相似度,相似度高的词,其"值(V)"就被更多地吸收进来。
# 这就是你第3周学的点积衡量相似度在实战中的应用。除以 √d_k 是为了防止点积数值过大,导致softmax饱和、梯度消失。
