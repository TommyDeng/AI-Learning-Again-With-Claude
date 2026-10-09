import math

import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(42)

# ===== 1. 准备数据 =====
text = (
    "machine learning is a field of artificial intelligence. "
    "deep learning uses neural networks with many layers. "
    "a transformer uses attention to understand context. "
    "language models learn to predict the next word. "
) * 30

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text))
block_size = 32  # 模型一次最多看32个字符
batch_size = 32


def get_batch():
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # 目标=输入右移一位
    return x, y


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


# ===== 2. 定义模型 =====
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, block_size=32):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        # 因果掩码:每个位置只能看到自己和之前的字符,不能偷看未来
        mask = torch.tril(torch.ones(T, T, device=idx.device))
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # 装饰器 将函数重新包装了一下 相当于:generate = torch.no_grad()(generate)。
    # 意思是generate之前，先执行torch.no_grad(),等 generate 函数执行完毕返回时，它又会自动恢复之前的梯度计算状态
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ===== 3. 训练 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(vocab_size, block_size=block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
print("参数量:", sum(p.numel() for p in model.parameters()))

for step in range(1500):
    x, y = get_batch()
    x, y = x.to(device), y.to(device)
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"step {step}: loss = {loss.item():.4f}")

# ===== 4. 生成文本 =====
model.eval()
start = torch.tensor([encode("a transformer")], device=device)
out = model.generate(start, max_new_tokens=150, temperature=0.8)
print("\n生成结果:\n", decode(out[0].tolist()))
