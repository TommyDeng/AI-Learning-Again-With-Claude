import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
# 这是社区前人通过遍历 CIFAR-10 的 50,000 张训练集图片，将每个像素点从 [0, 255] 缩放到 [0.0, 1.0] 后，精确计算出的统计结果。
# 在处理 CIFAR-10 数据集时，直接套用这组官方/通用统计值即可。
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # Windows下显示中文
plt.rcParams["axes.unicode_minus"] = False


# ==================== 1. 数据 ====================
def get_loaders(use_aug=True, batch_size=128, num_workers=2):
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    if use_aug:
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),  # 随机裁剪(先补边再裁)
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ColorJitter(0.2, 0.2, 0.2),  # 轻微颜色扰动
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    else:
        train_tf = test_tf

    # 同一份训练数据用两种transform:训练集带增强,验证集不带
    full_train_aug = datasets.CIFAR10(
        "./data", train=True, download=True, transform=train_tf
    )
    full_train_clean = datasets.CIFAR10(
        "./data", train=True, download=True, transform=test_tf
    )
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(50000, generator=g).tolist()
    train_idx, val_idx = perm[:45000], perm[45000:]  # 45000训练 / 5000验证

    train_loader = DataLoader(
        Subset(full_train_aug, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(full_train_clean, val_idx),
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


# ==================== 2. 模型:小型ResNet ====================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            # 尺寸或通道数变了,shortcut也要用1x1卷积调整,才能和主路径相加
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(
            out + self.shortcut(x)
        )  # 残差连接:和第13周Transformer里的 x + ... 同一思想


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32), ResidualBlock(32, 32)
        )  # 32x32
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, 2), ResidualBlock(64, 64)
        )  # 16x16
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, 2), ResidualBlock(128, 128)
        )  # 8x8
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化:每个通道压成一个数
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


# ==================== 3. 训练与评估 ====================
def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()  # OneCycleLR 要每个batch更新一次
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train(use_aug=True, epochs=15, patience=5, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_loaders(use_aug=use_aug)

    model = SmallResNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑:轻微抑制过度自信
    optimizer = optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-2
    )  # weight_decay就是第8周的L2正则
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, bad_epochs = 0.0, 0
    best_state = None

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler
        )
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(
            f"epoch {epoch + 1:2d}/{epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time() - t0:.0f}s"
        )

        # 保存验证集上最好的模型,并做早停判断
        if va_acc > best_val_acc:
            best_val_acc, bad_epochs = va_acc, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(best_state, "models/best_cifar10.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"验证集准确率连续{patience}轮没有提升,早停")
                break

    model.load_state_dict(best_state)
    return model, history


# ==================== 4. 结果分析 ====================
def plot_history(history, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="训练")
    axes[0].plot(history["val_loss"], label="验证")
    axes[0].set_title(f"{title} Loss")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="训练")
    axes[1].plot(history["val_acc"], label="验证")
    axes[1].set_title(f"{title} 准确率")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def evaluate_on_test(model, device):
    _, _, test_loader = get_loaders()
    model.eval()
    all_pred, all_true, all_images = [], [], []
    for images, labels in test_loader:
        preds = model(images.to(device)).argmax(1).cpu()
        all_pred.append(preds)
        all_true.append(labels)
        all_images.append(images)
    return (
        torch.cat(all_pred).numpy(),
        torch.cat(all_true).numpy(),
        torch.cat(all_images),
    )


def analyze(model, device):
    y_pred, y_true, images = evaluate_on_test(model, device)
    print(f"\n测试集准确率: {(y_pred == y_true).mean():.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # 混淆矩阵:看哪些类别容易被搞混(猫↔狗 通常最严重)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES
    )
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.title("混淆矩阵")
    plt.tight_layout()
    plt.show()

    # 展示一些预测错误的样本
    wrong = np.where(y_pred != y_true)[0][:10]
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, i in zip(axes.flat, wrong):
        img = (
            (images[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        )  # 反归一化再显示
        ax.imshow(img)
        ax.set_title(
            f"预测:{CLASSES[y_pred[i]]}\n真实:{CLASSES[y_true[i]]}",
            color="red",
            fontsize=10,
        )
        ax.axis("off")
    plt.suptitle("预测错误的样本")
    plt.tight_layout()
    plt.show()


# ==================== 主程序 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    model_aug, hist_aug = train(use_aug=True, epochs=15, device=device)
    model_raw, hist_raw = train(use_aug=False, epochs=15, device=device)

    plt.plot(hist_aug["val_acc"], label="有数据增强")
    plt.plot(hist_raw["val_acc"], label="无数据增强")
    plt.xlabel("epoch")
    plt.ylabel("验证集准确率")
    plt.legend()
    plt.title("数据增强的效果对比")
    plt.show()

    # 你会看到无增强的版本训练准确率冲得很高,但验证准确率更低、两者差距(过拟合)更大。
    print("有增强 训练/验证准确率:", hist_aug["train_acc"][-1], hist_aug["val_acc"][-1])
    print("无增强 训练/验证准确率:", hist_raw["train_acc"][-1], hist_raw["val_acc"][-1])

    plot_history(hist_aug, "带数据增强")
    analyze(model_aug, device)
