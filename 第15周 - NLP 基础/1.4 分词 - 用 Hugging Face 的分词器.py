from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-chinese")

text = "我喜欢学习人工智能"
encoded = tok(text)
print("input_ids:", encoded["input_ids"])
print("tokens:", tok.convert_ids_to_tokens(encoded["input_ids"]))
# 注意首尾的 [CLS] 和 [SEP],这是 BERT 的特殊标记

# 批处理时,不同长度的句子要补齐(padding)
batch = tok(["我爱AI", "今天天气很好,适合出去走走"], padding=True, return_tensors="pt")
print(batch["input_ids"].shape)
print(batch["attention_mask"])  # 1=真实token, 0=补齐的部分,注意力要忽略它
