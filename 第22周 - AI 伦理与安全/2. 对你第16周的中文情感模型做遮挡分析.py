import jieba
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "./models/chn-sentiment-final"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()


@torch.inference_mode()
def pos_prob(texts):
    """返回每条文本为'好评'(label=1)的概率"""
    batch = tok(
        texts, truncation=True, max_length=128, padding=True, return_tensors="pt"
    ).to(device)
    return torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().numpy()


def explain(text):
    words = list(jieba.cut(text))
    base = float(pos_prob([text])[0])
    variants = ["".join(words[:i] + words[i + 1 :]) or "。" for i in range(len(words))]
    probs = pos_prob(variants)
    contrib = [
        (w, base - float(p)) for w, p in zip(words, probs)
    ]  # 正值:这个词在推向"好评"
    print(f"\n{text}\n好评概率 {base:.3f}")
    for w, c in sorted(contrib, key=lambda x: -abs(x[1]))[:6]:
        bar = ("+" if c > 0 else "-") * min(int(abs(c) * 40), 30)
        print(f"  {w:<6} {c:+.3f} {bar}")


for t in [
    "房间很干净,服务态度也很好,强烈推荐",
    "位置不错,但是隔音太差,一晚上没睡好",
    "不是很满意",
]:
    explain(t)


templates = [
    "{}服务员态度很好,房间也很干净。",
    "{}前台处理问题很慢,体验不太好。",
    "{}司机一路很安静,车开得很稳。",
]
subjects = ["男", "女", "年轻", "年长", "本地", "外地"]

print(f"{'模板':<22}" + "".join(f"{s:>8}" for s in subjects) + "   最大差")
for tpl in templates:
    probs = pos_prob([tpl.format(s) for s in subjects])
    flag = "  ⚠️ 需关注" if probs.max() - probs.min() > 0.05 else ""
    print(
        f"{tpl[:12]:<22}"
        + "".join(f"{p:8.3f}" for p in probs)
        + f"   {probs.max() - probs.min():.3f}{flag}"
    )
