from collections import Counter


def get_pair_counts(words):
    pairs = Counter()
    for word, freq in words.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge_pair(words, pair):
    new_words = {}
    for word, freq in words.items():
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(word[i] + word[i + 1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        new_words[tuple(out)] = freq
    return new_words


corpus = "low low low low low lower lower newest newest newest newest widest widest"
words = Counter(tuple(w) for w in corpus.split())  # 每个词先拆成字符

for step in range(12):
    pairs = get_pair_counts(words)
    best = pairs.most_common(1)[0][0]
    words = merge_pair(words, best)
    print(f"第{step + 1}次合并: {best} -> {''.join(best)}")

print("\n最终切分:", dict(words))
