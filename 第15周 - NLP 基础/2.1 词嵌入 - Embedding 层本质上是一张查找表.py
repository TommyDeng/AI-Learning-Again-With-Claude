import torch
from torch import nn

emb = nn.Embedding(num_embeddings=10, embedding_dim=4)  # 词表10个词,每个词4维向量
ids = torch.tensor([1, 5, 1])
print(emb(ids))  # 第0行和第2行完全相同,因为是同一个词(id=1)
print(emb.weight.shape)  # [10, 4],这张表本身就是可学习的参数
