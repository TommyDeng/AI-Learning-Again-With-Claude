import json
import os

import anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_TYPE = os.getenv("API_TYPE", "GROQ").upper()
client = None
MODEL = None

if MODEL_TYPE == "ANTHROPIC":
    client = anthropic.Anthropic()  # 自动读取环境变量 ANTHROPIC_API_KEY
    MODEL = os.getenv("ANTHROPIC_MODEL")  # 模型名称更新很快,以官方文档为准
else:
    MODEL = os.getenv("GROQ_MODEL")
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("GROQ_URL"),
    )


# 这份代码无法正常使用，除非购买了Anthropic的API服务。
# 使用Groq提供的免费服务时，在第一步调用大模型后，无法给出使用工具的提示：stop_reason='stop',tool_calls=None

# 标准的工具定义格式（Schema）
# 固定要素：
# name：工具的唯一标识名（模型在决定调用时会直接返回这个名字）。
# description：告诉模型这个工具是干什么的、什么时候该用它（提示词工程的一部分，写得好坏直接决定模型会不会用）。
# parameters / input_schema：定义这个工具接收什么参数、每个参数的类型和含义是什么。
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市当前的气温(摄氏度)。只在需要天气信息时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名,例如 台北"}
                },
                "required": ["city"],
            },
        },
    }
]

FAKE_DB = {"台北": 26, "高雄": 31}

messages = [{"role": "user", "content": "台北和高雄现在哪个更热?差几度?"}]

# ---------- 第一次调用:模型决定要用工具 ----------
resp = client.chat.completions.create(
    model=MODEL, max_tokens=1000, tools=tools, messages=messages
)

choice = resp.choices[0]

print("stop_reason:", choice.finish_reason)  # 'tool_use' 表示模型在等你执行工具
print(resp)
if choice.message.content:
    print("文本:", choice.message.content)

tool_calls = choice.message.tool_calls
if tool_calls:
    for tc in tool_calls:
        print("工具调用:", tc.function.name, tc.function.arguments, "id =", tc.id)

# ---------- 执行工具,把结果还给模型 ----------
messages.append(choice.message)


for tc in tool_calls:
    # OpenAI 的 arguments 是 JSON 字符串，需要先用 json.loads 解析
    args = json.loads(tc.function.arguments)
    city = args.get("city")

    temp = FAKE_DB.get(city)
    tool_output = f"{temp}°C" if temp is not None else "查无此城市"

    # OpenAI 规范：工具结果必须以 role="tool" 的形式传回，并带上对应的 tool_call_id
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_output})


# ---------- 第二次调用:模型基于工具结果给出最终回答 ----------
resp2 = client.chat.completions.create(
    model=MODEL, max_tokens=1000, tools=tools, messages=messages
)

choice2 = resp2.choices[0]
print("\nfinish_reason:", choice2.finish_reason)  # 'stop' 表示模型任务完成
print("最终回答:", choice2.message.content)
