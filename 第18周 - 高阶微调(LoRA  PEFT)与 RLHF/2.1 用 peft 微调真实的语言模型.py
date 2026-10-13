# 任务设计

# 任务:从一句中文自我介绍里提取信息,严格输出固定格式的 JSON。

# 选这个任务的原因:

# 数据可以用代码批量生成,不依赖外部数据集(避免上周遇到的下载问题)
# 结果可以自动评分(JSON 能否解析、字段对不对),微调前后效果一目了然

# 模型用 Qwen/Qwen2.5-0.5B-Instruct(5 亿参数,中文能力不错,个人电脑能跑)。


import json
import random

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    torch.bfloat16
    if (device == "cuda" and torch.cuda.is_bf16_supported())
    else torch.float32
)

NAMES = [
    "小明",
    "小红",
    "王大伟",
    "陈美玲",
    "林志豪",
    "张雅婷",
    "李建国",
    "黄思琪",
    "吴宗翰",
    "周怡君",
]
CITIES = [
    "台北",
    "高雄",
    "台中",
    "新竹",
    "台南",
    "桃园",
    "上海",
    "北京",
    "深圳",
    "成都",
]
JOBS = [
    "工程师",
    "老师",
    "医生",
    "设计师",
    "会计",
    "律师",
    "厨师",
    "记者",
    "程序员",
    "销售",
]
TEMPLATES = [
    "我叫{name},今年{age}岁,住在{city},是一名{job}。",
    "大家好,我是{name},{age}岁,目前在{city}工作,职业是{job}。",
    "{name}今年{age}岁,来自{city},从事{job}工作。",
    "自我介绍一下:{job}一枚,名字叫{name},{age}岁,常驻{city}。",
]


def make_example(rng):
    f = dict(
        name=rng.choice(NAMES),
        age=rng.randint(18, 65),
        city=rng.choice(CITIES),
        job=rng.choice(JOBS),
    )
    text = rng.choice(TEMPLATES).format(**f)
    answer = json.dumps(
        f, ensure_ascii=False, separators=(",", ":")
    )  # 紧凑格式,字段顺序固定
    return text, answer, f


def build_sft_item(tok, text, answer, max_len=160):
    """把一条样本变成 input_ids + labels;只对回答部分计算 loss,提示部分设为 -100"""
    msgs = [{"role": "user", "content": f"提取信息:{text}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + answer + "<|im_end|>"  # Qwen 的结束符
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
    return {"input_ids": f_ids, "labels": labels[: len(f_ids)]}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


@torch.no_grad()
def generate(model, tok, text, max_new_tokens=60):
    msgs = [{"role": "user", "content": f"提取信息:{text}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def evaluate(model, tok, tests, label):
    model.eval()
    valid, exact = 0, 0
    for text, _, gold in tests:
        out = generate(model, tok, text)
        try:
            pred = json.loads(out)
            valid += 1
            exact += pred == gold
        except json.JSONDecodeError:
            pass
    n = len(tests)
    print(f"[{label}] JSON合法率 {valid / n:.0%} | 字段完全正确率 {exact / n:.0%}")


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device)

    # 数据:训练300条,测试40条(用不同的随机种子,模板相同但具体取值组合不同)
    train_raw = [make_example(rng) for _ in range(300)]
    test_raw = [make_example(random.Random(999)) for _ in range(40)]
    train_items = [build_sft_item(tok, t, a) for t, a, _ in train_raw]

    # ---------- 微调前 ----------
    print("微调前的一个例子:")
    print("  输入:", test_raw[0][0])
    print("  输出:", generate(model, tok, test_raw[0][0]))
    evaluate(model, tok, test_raw, "微调前")

    # ---------- 加 LoRA ----------
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],  # 给注意力的四个投影矩阵加 LoRA
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()  # 通常只有约 0.5%~1% 的参数可训练

    # ---------- 训练(和第10、16周是同一个循环) ----------
    loader = DataLoader(
        train_items,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    model.train()
    for epoch in range(3):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss  # 传入 labels,内部自动算交叉熵
            loss.backward()
            opt.step()
            opt.zero_grad()
            if step % 10 == 0:
                print(
                    f"epoch {epoch + 1} step {step}/{len(loader)} loss={loss.item():.4f}"
                )

    # ---------- 微调后 ----------
    print("\n微调后的同一个例子:")
    print("  输出:", generate(model, tok, test_raw[0][0]))
    evaluate(model, tok, test_raw, "微调后(LoRA)")

    # 对比:临时关掉 adapter,又回到原模型
    with model.disable_adapter():
        evaluate(model, tok, test_raw, "关闭adapter")

    # ---------- 保存:只保存 adapter ----------
    model.save_pretrained("./models/json-lora")
    print("\nadapter 已保存到 ./models/json-lora")


if __name__ == "__main__":
    main()
