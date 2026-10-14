import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
Z = 64


def make_models():
    G = nn.Sequential(
        nn.Linear(Z, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 784),
        nn.Tanh(),
    )  # 输出范围 [-1, 1]
    D = nn.Sequential(
        nn.Linear(784, 512),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )  # 输出 logit,不加 sigmoid
    return G.to(device), D.to(device)


def main():
    torch.manual_seed(0)
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )  # 缩放到 [-1,1]
    loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tf),
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )

    G, D = make_models()
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(16, Z, device=device)

    for epoch in range(20):
        for real, _ in loader:
            real = real.view(real.size(0), -1).to(device)
            b = real.size(0)
            ones, zeros = (
                torch.ones(b, 1, device=device),
                torch.zeros(b, 1, device=device),
            )

            # ---- 训练判别器 ----
            fake = G(torch.randn(b, Z, device=device)).detach()  # detach:此步不更新 G
            lossD = bce(D(real), ones) + bce(D(fake), zeros)
            optD.zero_grad()
            lossD.backward()
            optD.step()

            # ---- 训练生成器 ----
            fake = G(torch.randn(b, Z, device=device))
            lossG = bce(D(fake), ones)  # 让 D 把假图判成真
            optG.zero_grad()
            lossG.backward()
            optG.step()

        print(
            f"epoch {epoch + 1}: D loss {lossD.item():.3f}  G loss {lossG.item():.3f}"
        )

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                imgs = G(fixed_z).view(-1, 28, 28).cpu()
            fig, axes = plt.subplots(2, 8, figsize=(12, 3))
            for ax, im in zip(axes.flat, imgs):
                ax.imshow(im, cmap="gray")
                ax.axis("off")
            plt.suptitle(f"GAN 生成结果(epoch {epoch + 1})")
            plt.show()


if __name__ == "__main__":
    main()

# 为什么 GAN 难训练:

# 博弈不一定收敛: D 太强,G 得不到有用的梯度;D 太弱,G 学不到东西。看到两个 loss 剧烈震荡是常态,loss 数值不能直接代表图像质量,要看生成的图
# 模式坍塌(mode collapse): G 发现只生成某几种图就能骗过 D,于是失去多样性。比如只会生成"1"和"7"
# 这些问题推动了 WGAN、谱归一化等改进,也是后来大家转向扩散模型的重要原因
