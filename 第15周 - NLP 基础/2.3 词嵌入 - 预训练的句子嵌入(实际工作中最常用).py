import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # 支持中英文

sentences = [
    "我今天心情很好",
    "我今天非常开心",
    "这家餐厅的牛肉面很好吃",
    "The weather is lovely today",
    "今天天气真不错",
]
vecs = model.encode(sentences)
print("向量维度:", vecs.shape)

sim = cosine_similarity(vecs)
for i, s in enumerate(sentences):
    print(f"\n{s}")
    for j in np.argsort(-sim[i])[1:3]:
        print(f"   最相似: {sentences[j]}  ({sim[i][j]:.3f})")

# 你会看到语义相近的句子相似度高,即使字面上完全不同,甚至跨语言。这就是第18周 RAG 和向量数据库的基础。
