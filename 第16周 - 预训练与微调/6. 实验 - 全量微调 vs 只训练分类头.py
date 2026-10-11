from transformers import AutoModelForSequenceClassification

MODEL = "distilbert-base-uncased"


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model_full = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
print("全量微调 可训练参数:", f"{count_trainable(model_full):,}")

model_head = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
for name, p in model_head.named_parameters():
    if not name.startswith(("classifier", "pre_classifier")):  # DistilBERT 的分类头名字
        p.requires_grad = False
print("只训练分类头 可训练参数:", f"{count_trainable(model_head):,}")


# 用同样的数据和训练设置各跑一次,对比准确率和耗时,你会发现:

#               可训练参数	    速度/显存	    效果
# 全量微调	    ~6600 万	    慢、占显存	    通常最好
# 只训练分类头	~60 万	        快、省显存	    稍差,但数据极少时更稳

# 这个"参数量与效果的取舍",正是第19周 LoRA / PEFT 要解决的问题:只训练极少量新增参数,却接近全量微调的效果。
