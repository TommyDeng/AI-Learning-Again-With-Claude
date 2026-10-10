# 用 IMDB 影评数据集(英文,正面/负面二分类),对比传统方法 → 词嵌入 → 微调式思路,感受 NLP 技术的演进。


import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 取子集,保证普通电脑也能很快跑完
ds = load_dataset("stanfordnlp/imdb")
train = ds["train"].shuffle(seed=0).select(range(2000))
test = ds["test"].shuffle(seed=0).select(range(1000))

X_train_text, y_train = train["text"], np.array(train["label"])
X_test_text, y_test = test["text"], np.array(test["label"])

# ---------- 方法1:TF-IDF + 逻辑回归(传统基线) ----------
print("-" * 30)
print("方法1(TF-IDF):约 85%~88%,只统计词频,不懂语序和语义")
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
Xtr = tfidf.fit_transform(X_train_text)
Xte = tfidf.transform(X_test_text)

lr = LogisticRegression(max_iter=1000)
lr.fit(Xtr, y_train)
print("方法1  TF-IDF + 逻辑回归 :", accuracy_score(y_test, lr.predict(Xte)))

# 看模型学到了哪些词最能代表正/负面
names = np.array(tfidf.get_feature_names_out())
order = np.argsort(lr.coef_[0])
print("  最负面的词:", list(names[order[:10]]))
print("  最正面的词:", list(names[order[-10:]]))

# ---------- 方法2:预训练句子嵌入 + 逻辑回归 ----------
print("-" * 30)
print("方法2(句子嵌入):约 80%~85%,理解语义但嵌入模型不是专为情感设计的")
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")
Etr = encoder.encode(X_train_text, batch_size=64, show_progress_bar=True)
Ete = encoder.encode(X_test_text, batch_size=64, show_progress_bar=True)

lr2 = LogisticRegression(max_iter=1000)
lr2.fit(Etr, y_train)
print("方法2  句子嵌入 + 逻辑回归:", accuracy_score(y_test, lr2.predict(Ete)))


# ---------- 方法3:直接用已经微调好的预训练模型 ----------
print("-" * 30)
print(
    "方法3(微调的预训练模型):约 88%~92%,既有大规模预训练的语言知识,又针对情感任务专门微调过"
)
from transformers import pipeline

clf = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512,
)

preds = clf(X_test_text[:300], batch_size=16)
y_pred3 = np.array([1 if p["label"] == "POSITIVE" else 0 for p in preds])
print("方法3  微调过的DistilBERT :", accuracy_score(y_test[:300], y_pred3))
