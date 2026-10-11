import math

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL = "distilbert-base-uncased"

# 显存不够(CUDA out of memory): 调小 per_device_train_batch_size 和 MAX_LEN
MAX_LEN = 256
per_device_train_batch_size = 16  # 每个 GPU 上的 batch size,如果显存不够就调小


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)

    # ---------- 数据 ----------
    ds = load_dataset("stanfordnlp/imdb")
    train = ds["train"].shuffle(seed=0).select(range(2000))
    test = ds["test"].shuffle(seed=0).select(range(1000))

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LEN)

    train = train.map(tokenize, batched=True)
    test = test.map(tokenize, batched=True)
    # 不在这里 padding,而是让 DataCollator 每个 batch 按最长句子动态补齐,更省算力
    collator = DataCollatorWithPadding(tok)

    # ---------- 模型 ----------
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    steps_per_epoch = math.ceil(len(train) / per_device_train_batch_size)
    total_steps = steps_per_epoch * 2  # 2 就是 num_train_epochs
    warmup_steps = int(0.1 * total_steps)  # 前10%的步数用于预热

    # ---------- 训练参数 ----------
    args = TrainingArguments(
        output_dir=".models/imdb-distilbert",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=20,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("\n微调后评估:", trainer.evaluate())

    trainer.save_model("models/imdb-distilbert-final")
    tok.save_pretrained("models/imdb-distilbert-final")


if __name__ == "__main__":
    main()

# 传统模型（如 TF-IDF + 逻辑回归）和 你写的小模型：像一个只背了孤立单词和词频的初学者，只能靠硬记关键词来猜结果。

# BERT 模型：像一个精通全人类语言的“翻译官”，它脑子里有完整的语法规则和词语关联网络。你只需要用少量的标注数据（微调）告诉它：“老板，我现在要做情感分类”，它就能瞬间调动它庞大的语言功底，完美胜任这项工作。
