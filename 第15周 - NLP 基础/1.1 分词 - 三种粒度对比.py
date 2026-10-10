text = "Transformers are amazing!"

# 字符级:词表小,但序列很长
print(list(text))

# 单词级:词表巨大,遇到没见过的词就没办法(OOV问题)
print(text.split())

# 中文没有空格,单词级分词需要专门的工具
import jieba

print(list(jieba.cut("我喜欢学习人工智能")))
