from collections import Counter

from llm import ask


def self_consistency(ask, question, n=7):
    answers = [
        ask(question + "\n只回答一个简短的答案,不要解释。", temperature=1.0).strip()
        for _ in range(n)
    ]
    top, cnt = Counter(answers).most_common(1)[0]
    return top, cnt / n, answers


for q in [
    "台湾海拔最高的山是哪一座?",
    "《雾中码头》这本书的作者是谁?",  # 虚构的书
    "1987年台北市有多少家便利商店?",
]:  # 极可能不知道的冷门数字
    top, agree, answers = self_consistency(ask, q)
    print(f"{q}\n   最常见答案: {top}   一致率: {agree:.0%}   全部: {answers}\n")
