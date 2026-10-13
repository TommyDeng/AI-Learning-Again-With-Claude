import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)

# 方式1:挂载 adapter(可以随时换不同任务的 adapter,共用同一个底座)
model = PeftModel.from_pretrained(base, "./models/json-lora")

# 方式2:合并进权重,得到一个普通模型,推理没有额外开销
merged = model.merge_and_unload()
merged.save_pretrained("./models/json-merged")
tok.save_pretrained("./models/json-merged")

# 看看 adapter 文件夹有多大(通常只有几 MB):
import os

for f in os.listdir("./models/json-lora"):
    print(f, os.path.getsize(os.path.join("./models/json-lora", f)) // 1024, "KB")


# 查看一个模型有哪些层名可以加 LoRA:
for name, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        print(name)
        break
print(
    {
        n.split(".")[-1]
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
    }
)


# QLoRA(了解即可)

# QLoRA = 把冻结的底座模型量化成 4-bit + 在上面训练 LoRA。底座模型占的显存降到约 1/4,让消费级显卡也能微调 7B 甚至更大的模型。
# 需要 bitsandbytes 库,这个库对 Windows 的支持一直不太完整,以官方文档为准,Linux 或云端 GPU 环境最省心。
# 本周的 0.5B 模型不需要它。
