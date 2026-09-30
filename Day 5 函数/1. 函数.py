# 基本函数
def greet(name):
    return f"你好, {name}!"


print(greet("小明"))


# 默认参数
def power(base, exponent=2):
    return base**exponent


print(power(3))  # 9 (用默认值2)
print(power(3, 3))  # 27


# 关键字参数
def introduce(name, age, city="未知"):
    print(f"{name}, {age}岁, 来自{city}")


introduce(name="小红", age=22)
introduce("小李", 30, "台北")

# lambda匿名函数(常用于配合sorted/map/filter)
square = lambda x: x**2
print(square(5))  # 25
