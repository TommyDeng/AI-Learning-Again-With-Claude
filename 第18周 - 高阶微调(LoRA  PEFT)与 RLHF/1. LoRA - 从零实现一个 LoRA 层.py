import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False  # 冻结原权重
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # B 初始化为 0
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merge(self):
        """训练完把 ΔW 合并回原权重,推理时没有任何额外开销"""
        self.base.weight += (self.B @ self.A) * self.scale
        return self.base


# ---------- 验证1:初始状态下和原层输出完全一致 ----------
torch.manual_seed(0)
base = nn.Linear(512, 512)
lora = LoRALinear(base, r=8)
x = torch.randn(4, 512)
print("初始输出一致:", torch.allclose(base(x), lora(x)))  # True,因为 B=0

# ---------- 验证2:参数量对比 ----------
full = sum(p.numel() for p in base.parameters())
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(f"全量: {full:,}  LoRA: {trainable:,}  占比: {trainable / full:.2%}")


# 实验:秩 r 到底决定了什么
# 在实际工程中，大家通常会从 r = 8 或 r = 16 开始尝试。如果任务简单，甚至 r = 4 就足够了；
# 如果任务非常复杂（例如让小模型去学复杂的推理或多语言代码），才会考虑开到 32 或 64。
# 构造一个"真实增量恰好是秩 4"的任务,看不同的 r 能不能学出来:
torch.manual_seed(0)
d, true_rank = 64, 4
W0 = torch.randn(d, d)
delta = (torch.randn(d, true_rank) @ torch.randn(true_rank, d)) * 0.3  # 秩为4的增量
W_target = W0 + delta

X = torch.randn(2000, d)
Y = X @ W_target.T


def fit(r, steps=1500):
    base = nn.Linear(d, d, bias=False)
    base.weight.data = W0.clone()
    layer = LoRALinear(base, r=r, alpha=r)  # alpha=r 时 scale=1
    opt = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(steps):
        loss = F.mse_loss(layer(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


for r in [1, 2, 4, 8]:
    print(f"r={r}: 最终 loss = {fit(r):.5f}")
