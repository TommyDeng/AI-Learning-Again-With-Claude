import numpy as np

# 概率分布
data = np.random.normal(loc=0, scale=1, size=1000)  # 正态分布采样
print("均值:", np.mean(data))
print("标准差:", np.std(data))
print("方差:", np.var(data))

# 贝叶斯定理示例:医学检测问题
# P(患病)=0.01, P(阳性|患病)=0.99, P(阳性|未患病)=0.05
p_disease = 0.01
p_pos_given_disease = 0.99
p_pos_given_healthy = 0.05

p_pos = p_pos_given_disease * p_disease + p_pos_given_healthy * (1 - p_disease)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos

print(f"检测阳性后真正患病的概率: {p_disease_given_pos:.2%}")
# 结果会让你意识到:即使检测很准,患病概率低时,阳性结果也不代表大概率患病
