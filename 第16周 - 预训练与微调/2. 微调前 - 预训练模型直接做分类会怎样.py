# 分类需要在 BERT 上接一个新的分类头(一个随机初始化的线性层)。没微调之前,这个头什么都不懂:

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL)

ds = load_dataset("stanfordnlp/imdb")
test = ds["test"].shuffle(seed=0).select(range(300))

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
# 会看到警告 "Some weights ... newly initialized":这正是新加的分类头,警告是正常的
model.eval()

preds = []
with torch.no_grad():
    for i in range(0, len(test), 16):
        batch = tok(
            test["text"][i : i + 16],
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        preds += model(**batch).logits.argmax(-1).tolist()

print(
    "微调前准确率:", np.mean(np.array(preds) == np.array(test["label"]))
)  # 约50%,等于瞎猜
