import re

PATTERNS = {
    "手机号": re.compile(r"(?<!\d)09\d{2}[- ]?\d{3}[- ]?\d{3}(?!\d)"),  # 台湾手机号
    "邮箱": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "身份证号": re.compile(
        r"(?<![A-Za-z0-9])[A-Z][12]\d{8}(?!\d)"
    ),  # 台湾身分证字号格式
}


def mask_pii(text):
    found = {}
    for name, pat in PATTERNS.items():
        hits = pat.findall(text)
        if hits:
            found[name] = len(hits)
        text = pat.sub(f"[{name}]", text)
    return text, found


for s in [
    "我是王小明,电话 0912-345-678,邮箱 xiaoming.wang@example.com",
    "订单 A20931 的收货人手机 0987654321",
    "编号 123456789012 不是电话",
    "身份证 A123456789,订单 A20931",
]:
    print(mask_pii(s))


import numpy as np

rng = np.random.default_rng(0)
ages = rng.integers(18, 70, 5000)
true_count = int((ages > 40).sum())


def laplace_count(count, epsilon, sensitivity=1.0):
    # 计数查询的敏感度是 1:增删一个人,结果最多变化 1
    return count + rng.laplace(0, sensitivity / epsilon)


print("真实人数:", true_count)
for eps in [0.01, 0.1, 1.0, 10.0]:
    errs = [abs(laplace_count(true_count, eps) - true_count) for _ in range(2000)]
    print(f"epsilon={eps:<5} 平均绝对误差={np.mean(errs):8.2f}")
