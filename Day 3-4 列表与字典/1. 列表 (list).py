# 创建列表
fruits = ["苹果", "香蕉", "橙子"]

# 索引访问(从0开始)
print(fruits[0])  # 苹果
print(fruits[-1])  # 橙子 (负数从末尾数)

# 切片
print(fruits[0:2])  # ['苹果', '香蕉']

# 增删改
fruits.append("葡萄")  # 末尾添加
fruits.insert(1, "西瓜")  # 指定位置插入
fruits.remove("香蕉")  # 按值删除
fruits[0] = "草莓"  # 按索引修改

print(fruits)
print(len(fruits))  # 列表长度

# 列表推导式(非常重要,后面数据处理天天用)
numbers = [1, 2, 3, 4, 5]
squares = [n**2 for n in numbers]
print(squares)  # [1, 4, 9, 16, 25]

evens = [n for n in numbers if n % 2 == 0]
print(evens)  # [2, 4]
