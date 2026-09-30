# 算术运算符
a = 10
b = 3
print(a + b)  # 13  加
print(a - b)  # 7   减
print(a * b)  # 30  乘
print(a / b)  # 3.333...  除(结果是浮点数)
print(a // b)  # 3   整除(向下取整)
print(a % b)  # 1   取余(这个在很多算法题里很常用)
print(a**b)  # 1000  幂运算

# 比较运算符
print(a > b)  # True
print(a == b)  # False (注意:判断相等用两个等号!)

# 逻辑运算符
print(a > 5 and b > 5)  # False (且)
print(a > 5 or b > 5)  # True  (或)
print(not a > 5)  # False (非)

# 常见新手错误: 把赋值 = 和判断相等 == 搞混。x = 5 是赋值,x == 5 是判断。
