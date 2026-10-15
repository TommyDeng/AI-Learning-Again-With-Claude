import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)  # 对"输入"求梯度,而不是对参数
    return (x + eps * grad.sign()).clamp(0, 1).detach()  # 像素必须保持在合法范围


def pgd(model, x, y, eps, steps=10):
    """多步版本,攻击更强:每步走一小步,并把总扰动限制在 eps 之内"""
    alpha = eps / 4
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        (grad,) = torch.autograd.grad(F.cross_entropy(model(x_adv), y), x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    return x_adv.detach()


def train(train_loader, adv_eps=0.0, epochs=3):
    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if adv_eps > 0:  # 对抗训练:把攻击样本也加进训练集
                x_adv = fgsm(model, x, y, adv_eps)
                x, y = torch.cat([x, x_adv]), torch.cat([y, y])
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def accuracy(model, loader, attack=None, eps=0.0):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if attack and eps > 0:
            x = attack(model, x, y, eps)
        with torch.no_grad():
            correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    torch.manual_seed(0)
    tf = transforms.ToTensor()  # 保持 0~1,方便限制扰动范围
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tf),
        batch_size=128,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, transform=tf), batch_size=256
    )

    std_model = train(train_loader)
    adv_model = train(train_loader, adv_eps=0.2)

    print(
        f"{'eps':<6}{'普通模型 FGSM':>16}{'普通模型 PGD':>16}{'对抗训练 FGSM':>16}{'对抗训练 PGD':>16}"
    )
    for eps in [0.0, 0.05, 0.1, 0.2, 0.3]:
        row = [
            accuracy(std_model, test_loader, fgsm, eps),
            accuracy(std_model, test_loader, pgd, eps),
            accuracy(adv_model, test_loader, fgsm, eps),
            accuracy(adv_model, test_loader, pgd, eps),
        ]
        print(f"{eps:<6}" + "".join(f"{a:>16.3f}" for a in row))

    # 可视化:原图 vs 攻击后
    x, y = next(iter(test_loader))
    x, y = x[:6].to(device), y[:6].to(device)
    x_adv = fgsm(std_model, x, y, 0.2)
    with torch.no_grad():
        p0, p1 = std_model(x).argmax(1).cpu(), std_model(x_adv).argmax(1).cpu()
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
    for i in range(6):
        axes[0, i].imshow(x[i].cpu().squeeze(), cmap="gray")
        axes[0, i].set_title(f"原图 预测{p0[i].item()}")
        axes[1, i].imshow(x_adv[i].cpu().squeeze(), cmap="gray")
        axes[1, i].set_title(f"攻击后 预测{p1[i].item()}")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
