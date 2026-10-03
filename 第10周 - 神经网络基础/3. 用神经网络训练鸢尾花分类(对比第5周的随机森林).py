import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

# 准备数据
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转成PyTorch张量(Tensor)—— 神经网络只能处理Tensor,不能直接用numpy数组
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)


# 定义模型
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = IrisNet()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类任务标准损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器,比纯SGD更常用

# 训练循环
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()  # 清空上一轮的梯度
    outputs = model(X_train_t)  # 前向传播
    loss = criterion(outputs, y_train_t)  # 计算损失
    loss.backward()  # 反向传播,自动算出所有参数的梯度
    optimizer.step()  # 根据梯度更新参数

    if epoch % 20 == 0:
        print(f"epoch {epoch}: loss = {loss.item():.4f}")

# 测试
model.eval()  # 切换到评估模式
with torch.no_grad():  # 不需要计算梯度,节省资源
    test_outputs = model(X_test_t)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_t).float().mean()
    print(f"\n测试集准确率: {accuracy.item():.4f}")


# 关键步骤拆解(这是所有PyTorch训练代码的固定套路，务必理解每一步在干嘛):

# optimizer.zero_grad() — 梯度默认会累加，所以每轮开始前要清零                           1.归零
# model(X_train_t) — 前向传播，算出预测值                                              2.预测
# criterion(outputs, y_train_t) — 对比预测值和真实标签，算出损失                        3.评估损失
# loss.backward() — 第4周你手推的链式法则，在这里被自动执行，算出每个参数的梯度            4.反向传播
# optimizer.step() — 第4周你手写的梯度下降，在这里被自动执行，更新参数                    5.更新参数

# 你会发现，神经网络训练的本质和你第4周手写线性回归梯度下降一模一样,只是：参数变多了、结构变复杂了、梯度计算交给PyTorch自动完成(叫自动微分，autograd)。
