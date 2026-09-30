# for循环 — 遍历一个范围或集合
for i in range(5):
    print(i)  # 依次打印 0 1 2 3 4

# range(5) 生成 0,1,2,3,4 (不包含5)
# range(2, 5) 生成 2,3,4
# range(0, 10, 2) 生成 0,2,4,6,8 (步长为2)

# while循环 — 满足条件就一直执行
count = 0
while count < 5:
    print(count)
    count += 1  # 等价于 count = count + 1

# 练习1(现在就试试):
# 写一个循环,打印1到100之间所有能被7整除的数。
