import math


def perplexity(probs_of_true_tokens):
    # 困惑度 = exp(平均负对数似然),也就是 exp(交叉熵损失)
    nll = -sum(math.log(p) for p in probs_of_true_tokens) / len(probs_of_true_tokens)
    return math.exp(nll)


print("模型很有把握且猜对:", perplexity([0.9, 0.8, 0.95, 0.85]))
print("模型几乎在乱猜:    ", perplexity([0.1, 0.05, 0.1, 0.08]))

# 直觉理解: 困惑度可以理解为"模型平均每一步在多少个等可能的候选词之间犹豫"。越低越好。
# 第13周你训练迷你 GPT 时看到的 loss,取 exp 就是困惑度。
