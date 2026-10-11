import json
import re

from llm import ask


def extract_json(text):
    """容错地从模型输出里取出 JSON(可能被 ```json 围栏包着)"""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


email = """你好,我是王小明,订单号 A20931,上周三收到的耳机左耳没有声音,
希望能尽快换货,我的电话是 0912-345-678。"""

prompt = f"""从邮件中提取信息,只输出一个 JSON 对象,不要任何其他文字。

字段:
- name: 客户姓名
- order_id: 订单号
- issue: 问题描述(一句话)
- request: "换货" | "退款" | "维修" | "其他"
- phone: 电话,没有则为 null

<email>
{email}
</email>"""

data = extract_json(ask(prompt, temperature=0))
print(data)
print(type(data), data["order_id"])


# 工程上要记住: 模型输出永远可能出错。生产代码要 try/except 捕获解析失败,必要时重试,还要校验字段是否齐全、取值是否合法:


def extract_with_retry(prompt, required, retries=2):
    for attempt in range(retries + 1):
        try:
            data = extract_json(ask(prompt, temperature=0))
            missing = [k for k in required if k not in data]
            if not missing:
                return data
            print(f"第{attempt + 1}次缺少字段 {missing},重试")
        except json.JSONDecodeError as e:
            print(f"第{attempt + 1}次 JSON 解析失败:{e}")
    return None


print(extract_with_retry(prompt, ["name", "order_id", "issue", "request", "phone"]))
