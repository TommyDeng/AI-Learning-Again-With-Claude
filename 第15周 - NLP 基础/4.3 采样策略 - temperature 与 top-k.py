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


def sample_next(logits, temperature=1.0, top_k=None):
    logits = logits / temperature
    if top_k:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


torch.manual_seed(0)
for t in [0.2, 1.0, 2.0]:
    picks = [
        tok.decode(sample_next(next_logits.clone(), temperature=t)) for _ in range(8)
    ]
    print(f"temperature={t}: {picks}")

# temperature 低 → 分布变尖锐,几乎总选最可能的词(稳定、保守、重复)
# temperature 高 → 分布变平坦,更多样但更容易胡言乱语
# top-k → 只在概率最高的 k 个词里采样,砍掉离谱的长尾
