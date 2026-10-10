import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
lm = AutoModelForCausalLM.from_pretrained("gpt2")
lm.eval()

prompt = "The capital of France is"
inputs = tok(prompt, return_tensors="pt")

with torch.no_grad():
    logits = lm(**inputs).logits  # [1, 序列长度, 词表大小50257]

next_logits = logits[0, -1]  # 只看最后一个位置:预测"下一个词"
probs = torch.softmax(next_logits, dim=-1)
top = torch.topk(probs, 8)

print(f"提示词: {prompt!r}\n下一个词的候选:")
for p, i in zip(top.values, top.indices):
    print(f"  {tok.decode(i)!r:12} {p.item():.4f}")
