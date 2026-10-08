import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 下载MNIST数据集(28x28手写数字图片,0-9共10类)
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 看一眼数据长什么样
import matplotlib.pyplot as plt

images, labels = next(iter(train_loader))
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(images[i].squeeze(), cmap="gray")
    axes[i].set_title(f"标签: {labels[i].item()}")
    axes[i].axis("off")
plt.show()


# 用上一节课定义的 SimpleCNN 训练
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 卷积层部分:提取特征

        # 输入1通道(灰度图)，输出16通道，卷积核尺寸3x3，padding=1保持尺寸不变
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        # 输入16通道，输出32通道，卷积核尺寸3x3，padding=1保持尺寸不变
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # 全连接层部分:分类
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 28x28图经过两次池化变成7x7
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 卷积+池化反复堆叠
        # 图片尺寸经过两次池化过程的变化： 28x28 -> 14x14 -> 7x7
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7

        # 展平成一维向量,送进全连接层，目的是为了做分类
        # 整个过程就像是把装在 32 个 $7 \times 7$ 盒子里的积木，按顺序倒出来排成了一长排，数据原本的数值一模一样，没有任何信息在物理上被删减。
        x = x.view(x.size(0), -1)

        # 再经过两层全连接层，输出分类结果
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 用上面定义的 SimpleCNN 训练
model = SimpleCNN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("使用设备:", device)

epochs = 3
for epoch in range(epochs):
    # 训练
    # 将模型设置为训练模式，此时Dropout会随机丢弃部分神经元, BN会使用当前batch的均值和方差
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch {epoch + 1}: 平均loss = {total_loss / len(train_loader):.4f}")

# 测试
# 将模型设置为评估模式，此时Dropout已经关闭, BN也会使用训练时的均值和方差, 以保证测试结果的稳定性
model.eval()
correct = 0
total = 0
with torch.no_grad():  # 关闭梯度计算,节省显存和计算资源
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\n测试集: {correct}/{total}")
print(f"\n测试集准确率: {correct / total:.4f}")


# 可视化一些预测结果
model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].cpu().squeeze(), cmap="gray")
    color = "green" if predicted[i] == labels[i] else "red"
    ax.set_title(f"预测:{predicted[i].item()} 真实:{labels[i].item()}", color=color)
    ax.axis("off")
plt.tight_layout()
plt.show()


# 跑完这个项目，你的CNN在手写数字识别上应该能达到98%以上的准确率，只用了3个epoch——这是深度学习在结构化视觉任务上威力的直接体现。
# 如果没有GPU，训练会慢一些但完全跑得动，MNIST数据集很小。


# 认识经典架构(了解即可，不需要从零实现)

from torchvision import models

# PyTorch自带预训练的经典CNN架构
resnet = models.resnet18(weights=None)  # weights='IMAGENET1K_V1' 可加载预训练权重
print(resnet)
