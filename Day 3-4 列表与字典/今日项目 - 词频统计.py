text = "the quick brown fox jumps over the lazy dog the fox runs"

words = text.split()  # 按空格分词

word_count = {}
for word in words:
    word_count[word] = word_count.get(word, 0) + 1

# 按出现次数排序,取前5
top_5 = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:5]

for word, count in top_5:
    print(f"{word}: {count}次")

# 这个小项目其实就是最原始的"词袋模型"雏形,后面学NLP时你会再见到这个思路。
