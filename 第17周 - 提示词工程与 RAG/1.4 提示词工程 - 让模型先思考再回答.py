import re

from llm import ask

question = "一个商店先涨价20%,然后打八折出售。最终价格比原价高还是低?差多少?"

# 直接问
print(ask(question + "\n只给出结论。", temperature=0))
print("-" * 40)

# 先推理再回答,并用标签把两部分分开,方便程序只取答案
prompt = f"""{question}

请先在 <thinking> 标签里一步步推理,再在 <answer> 标签里给出最终答案。
注意：上述的<thinking>标签和<answer>标签不代表任何特殊含义，代表只需直接输出纯文本标签和内容，也绝对不要调用任何外部工具或函数。"""
out = ask(prompt, temperature=0)
print(out)

answer = re.search(r"<answer>(.*?)</answer>", out, re.DOTALL)
print("\n解析出的答案:", answer.group(1).strip() if answer else "解析失败")


# 对需要多步推理的问题(数学、逻辑、复杂分析),让模型先写出推理过程通常更准,因为它是一个词一个词往下生成的,写出中间步骤就等于给自己留了"草稿纸"。
