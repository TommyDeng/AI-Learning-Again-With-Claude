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
elif MODEL_TYPE == "GROQ":
    MODEL = os.getenv("GROQ_MODEL")
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("GROQ_URL"),
    )
else:
    MODEL = os.getenv("OLLAMA_MODEL")
    client = OpenAI(
        api_key=os.getenv("OLLAMA_API_KEY"),
        base_url=os.getenv("OLLAMA_URL"),
    )


# 由于Anthropic API 是单独计费的，此处改为免费的GROQ的 API调用方式，使用GROQ的OpenAI兼容接口
# 加载当前目录下的 .env


def ask(prompt, system=None, max_tokens=2000, temperature=None):

    kwargs = {
        "max_tokens": max_tokens,
        "model": MODEL,
    }

    messages = [{"role": "user", "content": prompt}]
    if system:  # 非空字符串
        if MODEL_TYPE == "ANTHROPIC":
            kwargs["system"] = system
        else:
            messages.insert(0, {"role": "system", "content": system})
    kwargs["messages"] = messages

    if temperature is not None:  # 即使是0也要传入，所以使用 is not None 判断
        kwargs["temperature"] = temperature
    print(f"kwargs: {kwargs}")

    # 使用 try...except 捕获 API 调用过程中的错误并优雅显示
    try:
        if MODEL_TYPE == "ANTHROPIC":
            resp = client.messages.create(**kwargs)
            return resp.content[0].text
        else:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

    except anthropic.APIStatusError as e:
        # 专门捕获 Anthropic 的状态码错误（如 400 余额不足、401 密钥错误等）
        return f"[API 错误提示] 请求失败: {e.message} (错误码: {e.status_code})"


if __name__ == "__main__":
    print(ask("用一句话解释你是谁"))
