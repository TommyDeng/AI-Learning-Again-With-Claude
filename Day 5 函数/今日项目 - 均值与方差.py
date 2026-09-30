# 均值（Mean）和方差（Variance）是概率论与统计学中最基础、也最核心的两个概念。简单来说，它们一个是数据的“核心位置”，另一个是数据的“散乱程度”。
# 如果用一句话来概括它们的精髓：“均值看高低，方差看稳不稳”。


def mean(numbers):
    return sum(numbers) / len(numbers)


def variance(numbers):
    m = mean(numbers)
    return sum((x - m) ** 2 for x in numbers) / len(numbers)


data = [2, 4, 4, 4, 5, 5, 7, 9]
print("均值:", mean(data))
print("方差:", variance(data))
