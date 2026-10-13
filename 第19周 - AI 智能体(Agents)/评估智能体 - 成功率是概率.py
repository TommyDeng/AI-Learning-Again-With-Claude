import statistics

from agent import agent_turn
from tools import CONFIG, WORKSPACE

CONFIG["confirm_writes"] = False  # 自动评测时关掉人工确认(仅限沙箱内)


def file_has(name, *words):
    p = WORKSPACE / name
    return p.exists() and all(w in p.read_text(encoding="utf-8") for w in words)


def clean(ans):
    return ans.replace(",", "").replace(" ", "")


TASKS = [
    (
        "sales.csv 的总销售额(units*price 之和)是多少?只回答纯数字,不要千分位逗号。",
        lambda ans: "932000" in clean(ans),
        None,
    ),
    ("哪个产品销售额最高?只回答产品名。", lambda ans: "显示器" in ans, None),
    (
        "根据 sales.csv 把销售额排名写入 ranking.md,必须包含每个产品名。",
        lambda ans: file_has("ranking.md", "耳机", "键盘", "鼠标", "显示器"),
        "ranking.md",
    ),
    (
        "读取 not_exist.txt 并总结",
        lambda ans: any(w in ans for w in ["不存在", "找不到", "没有", "未找到"]),
        None,
    ),
]

RUNS = 3
for task, check, cleanup in TASKS:
    passed, steps, tokens = 0, [], []
    for _ in range(RUNS):
        if cleanup and (WORKSPACE / cleanup).exists():
            (WORKSPACE / cleanup).unlink()
        msgs = [{"role": "user", "content": task}]
        r = agent_turn(msgs, verbose=False)
        ok = r["finished"] and check(r["answer"])
        passed += ok
        steps.append(r["stats"]["steps"])
        tokens.append(r["stats"]["input_tokens"] + r["stats"]["output_tokens"])
    print(
        f"{passed}/{RUNS} 通过 | 平均步数 {statistics.mean(steps):.1f} | "
        f"平均token {statistics.mean(tokens):.0f} | {task[:30]}..."
    )

# 关于成本: 每一步都要把全部历史重发一遍,所以总 token 数大致随步数平方级增长。
# 这是智能体比普通问答贵得多的原因。控制成本的方法:精简工具输出、限制 max_steps、对长任务做历史摘要、能用固定流程的部分就别交给智能体。

# 检查器要写得宽容: 注意我在题目里要求"只回答纯数字,不要千分位",并在 clean 里去掉逗号和空格。
# 检查器过于死板,会把"答对了但格式不同"误判为失败。评测代码本身也会有 bug,失败时先看实际输出再下结论。
