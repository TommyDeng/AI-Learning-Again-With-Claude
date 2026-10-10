import random
from collections import Counter, defaultdict

text = """
i love deep learning . i love machine learning . i like deep neural networks .
machine learning is fun . deep learning is powerful . neural networks learn patterns .
"""
words = text.split()

# 统计"当前词 -> 下一个词"的频次,归一化就是条件概率
counts = defaultdict(Counter)
for a, b in zip(words, words[1:]):
    counts[a][b] += 1


def next_word_probs(word):
    total = sum(counts[word].values())
    return {w: c / total for w, c in counts[word].items()}


print("'deep' 之后的词概率:", next_word_probs("deep"))
print("'learning' 之后的词概率:", next_word_probs("learning"))
print("'love' 之后的词概率:", next_word_probs("love"))


def generate(start, n=10):
    out = [start]
    for _ in range(n):
        probs = next_word_probs(out[-1])
        out.append(random.choices(list(probs), weights=probs.values())[0])
    return " ".join(out)


random.seed(1)
print(generate("i"))

# Bigram 只看前一个词,所以生成的句子局部通顺但整体无逻辑。Transformer 的改进就是:能同时参考前面几千上万个词。
