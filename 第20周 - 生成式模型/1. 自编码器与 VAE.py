import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class VAE(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 400), nn.ReLU())
        self.fc_mu = nn.Linear(400, latent)
        self.fc_logvar = nn.Linear(400, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, 400), nn.ReLU(), nn.Linear(400, 784), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x.view(-1, 784))
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)  # 重参数化
        return self.dec(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    bce = F.binary_cross_entropy(recon, x.view(-1, 784), reduction="sum")
    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )  # 与 N(0,1) 的 KL 的闭式解
    return (bce + beta * kl) / x.size(0), bce / x.size(0), kl / x.size(0)


def main():
    torch.manual_seed(0)
    data = datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )
    loader = DataLoader(data, batch_size=128, shuffle=True)

    model = VAE(latent=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        tot = rec = kl_sum = 0
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, bce, kl = vae_loss(recon, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n = x.size(0)
            tot += loss.item() * n
            rec += bce.item() * n
            kl_sum += kl.item() * n
        N = len(data)
        print(
            f"epoch {epoch + 1}: loss {tot / N:.2f} = 重建 {rec / N:.2f} + KL {kl_sum / N:.2f}"
        )

    model.eval()

    # ---------- 图1:每个数字在2维潜在空间里的位置 ----------
    zs, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(data, batch_size=1000):
            mu, _ = model.encode(x.to(device))
            zs.append(mu.cpu())
            labels.append(y)
    zs, labels = torch.cat(zs), torch.cat(labels)
    plt.figure(figsize=(7, 6))
    plt.scatter(zs[:, 0], zs[:, 1], c=labels, cmap="tab10", s=2)
    plt.colorbar()
    plt.title("潜在空间:数字自动聚成了簇")
    plt.show()

    # ---------- 图2:在潜在空间里均匀取点,逐个解码 ----------
    n = 15
    grid = torch.linspace(-3, 3, n)
    canvas = torch.zeros(n * 28, n * 28)
    with torch.no_grad():
        for i, yv in enumerate(reversed(grid.tolist())):
            for j, xv in enumerate(grid.tolist()):
                z = torch.tensor([[xv, yv]], device=device)
                canvas[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = (
                    model.dec(z).view(28, 28).cpu()
                )
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")
    plt.title("潜在空间流形:一个数字平滑地变成另一个")
    plt.show()

    # ---------- 图3:重建对比 ----------
    x, _ = next(iter(DataLoader(data, batch_size=10, shuffle=True)))
    with torch.no_grad():
        recon, _, _ = model(x.to(device))
    fig, axes = plt.subplots(2, 10, figsize=(14, 3))
    for i in range(10):
        axes[0, i].imshow(x[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].view(28, 28).cpu(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("原图", fontsize=9)
    axes[1, 0].set_title("重建", fontsize=9)
    plt.show()


if __name__ == "__main__":
    main()
