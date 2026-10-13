from agent import agent_turn
from tools import CONFIG

CONFIG["confirm_writes"] = True


def chat():
    messages = []
    print("文档助手已启动(输入 q 退出,/new 开启新对话)")
    while True:
        user = input("\n你:").strip()
        if user.lower() in ("q", "quit", "exit"):
            break
        if user == "/new":
            messages = []
            print("已清空对话历史")
            continue
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        result = agent_turn(messages, max_steps=12)
        print(f"\n助手:{result['answer']}")
        s = result["stats"]
        print(
            f"   [本轮 {s['steps']} 步, {s['tool_calls']} 次工具调用, "
            f"{s['input_tokens'] + s['output_tokens']} tokens]"
        )


if __name__ == "__main__":
    chat()


# 试这样的连续对话,检验它是否真的有上下文:

# 你:workspace 里有哪些文件?
# 你:把 notes 里的待办事项整理出来
# 你:把刚才那些写成 todo.md      ← "刚才那些"要靠对话历史才能理解
# 你:再把 sales.csv 里卖得最好的产品加到 todo.md 末尾

# 历史会越来越长。 简单策略:不要在 tool_use 和对应的 tool_result 中间截断历史(它们必须成对出现,否则 API 会报错)。更稳妥的办法是对话太长时,让模型把前面的内容总结成一段摘要,然后用摘要开启新对话。
