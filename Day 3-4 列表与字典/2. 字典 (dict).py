# 创建字典
person = {"name": "小明", "age": 25, "city": "台北"}

# 访问
print(person["name"])  # 小明
print(person.get("job", "未知"))  # 用get避免key不存在报错

# 增删改
person["job"] = "工程师"  # 添加
person["age"] = 26  # 修改
del person["city"]  # 删除

# 遍历
for key, value in person.items():
    print(key, ":", value)

# 常用方法
print(person.keys())
print(person.values())
