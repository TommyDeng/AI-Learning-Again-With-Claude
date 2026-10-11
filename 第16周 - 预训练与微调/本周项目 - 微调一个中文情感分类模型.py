import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL = "hfl/chinese-roberta-wwm-ext"  # 针对中文预训练的 BERT 变体
MAX_LEN = 128


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds),
    }


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)

    ds = load_dataset("lansinuote/ChnSentiCorp")  # 酒店评论,label: 0差评 1好评
    print(ds)  # 先看看有哪些划分和列名
    train = ds["train"].shuffle(seed=0).select(range(2000))
    test = ds["test"]

    tokenize = lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN)
    train, test = train.map(tokenize, batched=True), test.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    args = TrainingArguments(
        output_dir="./chn-sentiment",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        warmup_steps=0.1,  # v5:小于1的小数 = 占总训练步数的比例=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # 详细评估 + 错例分析
    out = trainer.predict(test)
    preds = out.predictions.argmax(-1)
    print(classification_report(out.label_ids, preds, target_names=["差评", "好评"]))

    wrong = np.where(preds != out.label_ids)[0][:5]
    for i in wrong:
        print(
            f"\n真实:{'好评' if out.label_ids[i] else '差评'} 预测:{'好评' if preds[i] else '差评'}"
        )
        print(test[int(i)]["text"][:120])

    trainer.save_model("./models/chn-sentiment-final")
    tok.save_pretrained("./models/chn-sentiment-final")


if __name__ == "__main__":
    main()
