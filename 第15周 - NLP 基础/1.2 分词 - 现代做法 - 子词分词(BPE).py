import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 使用的分词器

for s in ["Hello world", "unbelievable", "我喜欢学习人工智能", "def hello_world():"]:
    ids = enc.encode(s)
    pieces = [enc.decode([i]) for i in ids]
    print(f"{s!r}\n  token数: {len(ids)}\n  ids: {ids}\n  切分: {pieces}\n")

# BPE 的核心思想: 从单个字符开始,不断把语料里出现频率最高的相邻符号对合并成新符号,直到词表达到预定大小。
# 这样既避免了词表爆炸,又没有 OOV 问题,任何文字最坏情况都能拆回字节。
