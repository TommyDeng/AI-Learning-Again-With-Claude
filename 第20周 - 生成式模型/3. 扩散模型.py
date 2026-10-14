import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
T = 300
betas = torch.linspace(1e-4, 0.04, T, device=device)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)


# ==================== 噪声网络:带时间嵌入的小型 U-Net ====================
class SinusoidalEmb(nn.Module):
    """和第13周的位置编码是同一个想法:把整数 t 变成一个向量"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, out_ch)
        self.n2 = nn.GroupNorm(8, out_ch)
        self.t_proj = nn.Linear(t_dim, out_ch)

    def forward(self, x, te):
        h = F.silu(self.n1(self.conv1(x)))
        h = h + self.t_proj(te)[:, :, None, None]  # 把"现在是第几步"注入特征图
        return F.silu(self.n2(self.conv2(h)))


class TinyUNet(nn.Module):
    def __init__(self, t_dim=128):
        super().__init__()
        self.temb = nn.Sequential(
            SinusoidalEmb(t_dim), nn.Linear(t_dim, t_dim), nn.SiLU()
        )
        self.d1 = Block(1, 32, t_dim)  # 28x28
        self.d2 = Block(32, 64, t_dim)  # 14x14
        self.mid = Block(64, 64, t_dim)  # 7x7
        self.u2 = Block(64 + 64, 32, t_dim)  # 14x14,拼接了 d2 的跳跃连接
        self.u1 = Block(32 + 32, 32, t_dim)  # 28x28,拼接了 d1 的跳跃连接
        self.out = nn.Conv2d(32, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        te = self.temb(t)
        h1 = self.d1(x, te)
        h2 = self.d2(self.pool(h1), te)
        m = self.mid(self.pool(h2), te)
        u = self.u2(torch.cat([F.interpolate(m, scale_factor=2), h2], 1), te)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), te)
        return self.out(u)  # 预测的噪声,形状与输入相同


# ==================== 前向加噪 / 反向采样 ====================
def q_sample(x0, t, noise):
    ab = alpha_bar[t][:, None, None, None]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


@torch.no_grad()
def sample(model, n=16):
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    for t in reversed(range(T)):
        tb = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, tb)
        mean = (x - betas[t] / (1 - alpha_bar[t]).sqrt() * eps) / alphas[t].sqrt()
        x = mean + betas[t].sqrt() * torch.randn_like(x) if t > 0 else mean
    model.train()
    return x.clamp(-1, 1)


def show(imgs, title):
    imgs = (imgs.cpu().squeeze(1) + 1) / 2  # 还原到 [0,1]
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for ax, im in zip(axes.flat, imgs):
        ax.imshow(im, cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.show()


def main():
    torch.manual_seed(0)
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    data = datasets.MNIST("./data", train=True, download=True, transform=tf)
    loader = DataLoader(data, batch_size=128, shuffle=True, drop_last=True)

    # ---------- 先看前向加噪:一张图如何逐渐变成噪声 ----------
    x0 = data[0][0].unsqueeze(0).to(device)
    steps = [0, 30, 60, 100, 150, 200, 299]
    fig, axes = plt.subplots(1, len(steps), figsize=(14, 2.2))
    for ax, s in zip(axes, steps):
        xt = q_sample(x0, torch.tensor([s], device=device), torch.randn_like(x0))
        ax.imshow(xt.cpu().squeeze(), cmap="gray")
        ax.set_title(f"t={s}")
        ax.axis("off")
    plt.suptitle("前向过程:逐步加噪")
    plt.show()

    # ---------- 训练 ----------
    model = TinyUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    print("参数量:", sum(p.numel() for p in model.parameters()))

    for epoch in range(8):
        total = 0
        for x0, _ in loader:
            x0 = x0.to(device)
            t = torch.randint(
                0, T, (x0.size(0),), device=device
            )  # 每张图随机抽一个时间步
            noise = torch.randn_like(x0)
            loss = F.mse_loss(model(q_sample(x0, t, noise), t), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss {total / len(loader):.4f}")
        if epoch in (0, 3, 7):
            show(sample(model), f"epoch {epoch + 1} 生成结果")

    torch.save(model.state_dict(), "./models/tiny_diffusion.pt")


if __name__ == "__main__":
    main()
# 运行提示:

# GPU 上每个 epoch 约几十秒;纯 CPU 会很慢(U-Net 要处理 28×28 的特征图,采样还要走 300 步)。
# CPU 上建议把 epoch 缩到 3、T 缩到 100(同时把 betas 末端改成 0.1 左右,保证最后接近纯噪声)

# 第1个 epoch 后的图通常只是模糊的团块,到第8个 epoch 应该能看出清晰的数字,但不会很完美。
# 我们的网络很小、训练很短,真实扩散模型要大几个数量级
# 采样时的 x = mean + σ·噪声 每一步都在去噪的同时重新注入一点随机性,最后一步(t=0)不再加噪
