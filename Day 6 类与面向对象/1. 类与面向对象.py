class Dog:
    def __init__(self, name, age):
        self.name = name  # 属性
        self.age = age

    def bark(self):  # 方法
        print(f"{self.name}: 汪汪!")

    def info(self):
        print(f"{self.name}今年{self.age}岁")


# 使用
my_dog = Dog("旺财", 3)
my_dog.bark()
my_dog.info()


# 继承(PyTorch模型定义会大量用到这个概念)
class Puppy(Dog):
    def __init__(self, name, age):
        super().__init__(name, age)  # 调用父类初始化

    def bark(self):  # 重写方法
        print(f"{self.name}: 汪！(奶声奶气)")


p = Puppy("小白", 0.5)
p.bark()


# 为什么类很重要: 后面用PyTorch搭神经网络时,写法是这样的——

# python
# import torch.nn as nn

# class MyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 定义层

#     def forward(self, x):
#         # 定义前向传播逻辑
#         return x

# 看起来眼熟了吧?这就是为什么现在要打好类的基础。
