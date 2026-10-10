# 核心想法:一个词的含义由它周围的词决定。让模型根据中心词预测上下文词,训练过程中词向量自然就带上了语义。
import torch
import torch.nn.functional as F
from torch import nn

corpus = """
the king rules the kingdom . the queen rules the kingdom .
the man walks in the city . the woman walks in the city .
the king is a man . the queen is a woman .
the prince is a young man . the princess is a young woman .
a dog chases a cat . a cat chases a mouse .
the dog is an animal . the cat is an animal . the mouse is an animal .
"""
tokens = corpus.split()
vocab = sorted(set(tokens))
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}

# 构造(中心词, 上下文词)训练对,窗口大小=2
window = 2
pairs = []
for i, w in enumerate(tokens):
    for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
        if i != j:
            pairs.append((w2i[w], w2i[tokens[j]]))

centers = torch.tensor([p[0] for p in pairs])
contexts = torch.tensor([p[1] for p in pairs])


class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x):
        return self.out(self.emb(x))


torch.manual_seed(0)
model = SkipGram(len(vocab), 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

for epoch in range(300):
    loss = F.cross_entropy(model(centers), contexts)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}")

# 余弦相似度:就是第3周学的点积,只是先做了归一化
E = F.normalize(model.emb.weight.detach(), dim=1)


def most_similar(word, k=4):
    sims = E @ E[w2i[word]]
    top = sims.argsort(descending=True)[1 : k + 1]
    return [(i2w[i.item()], round(sims[i].item(), 3)) for i in top]


for w in ["king", "woman", "dog"]:
    print(w, "->", most_similar(w))
