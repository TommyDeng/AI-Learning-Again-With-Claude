from transformers import pipeline

fill = pipeline("fill-mask", model="bert-base-uncased")
for r in fill("The capital of France is [MASK].")[:5]:
    print(f"{r['token_str']:12} {r['score']:.4f}")

print()
for r in fill("I loved this movie, it was absolutely [MASK].")[:5]:
    print(f"{r['token_str']:12} {r['score']:.4f}")

# BERT 预训练时从没人教过它"巴黎是首都"或"什么是好评",它只是玩了几十亿次完形填空,这些知识就被迫学进了参数里。
