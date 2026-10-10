from transformers import pipeline

# 情感分析
clf = pipeline(
    "sentiment-analysis", model="uer/roberta-base-finetuned-dianping-chinese"
)
print(clf(["这家店的服务太差了,再也不来了", "味道很棒,下次还会再来!"]))

# 文本生成
gen = pipeline("text-generation", model="gpt2")
print(
    gen(
        "Artificial intelligence will",
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
    )[0]["generated_text"]
)
