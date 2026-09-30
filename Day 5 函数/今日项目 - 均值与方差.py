def mean(numbers):
    return sum(numbers) / len(numbers)


def variance(numbers):
    m = mean(numbers)
    return sum((x - m) ** 2 for x in numbers) / len(numbers)


data = [2, 4, 4, 4, 5, 5, 7, 9]
print("均值:", mean(data))
print("方差:", variance(data))
