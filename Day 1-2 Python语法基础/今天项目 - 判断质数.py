# 今天的动手小项目:判断质数
# 结合今天学的if和循环,来写一个稍微有难度的练习:


def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False  # 找到能整除的数,说明不是质数
    return True


# 测试
for num in range(2, 30):
    if is_prime(num):
        print(num, "是质数")


# 完成练习1(打印1-100里能被7整除的数)
# 试着自己写一个"猜数字游戏":程序随机生成一个1-100的数,用户输入猜测,程序提示"大了"/"小了"/"猜对了"

# 猜数字游戏的提示: 需要用到 import random 和 random.randint(1, 100) 来生成随机数,以及 input() 来获取用户输入。
