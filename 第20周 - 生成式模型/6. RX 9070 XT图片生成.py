import gc
import os
import time

import torch
from diffusers import AutoPipelineForText2Image

# 可选:ROCm 环境下加快首次运行(必须在 import torch 前设置才最稳,可放到 shell 里)
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

os.makedirs("./data", exist_ok=True)

device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # ROCm 版 PyTorch 也使用 "cuda" 这个名字
use_gpu = device == "cuda"
print("使用设备:", torch.cuda.get_device_name(0) if use_gpu else "CPU")
print("HIP 版本:", getattr(torch.version, "hip", None))

PROMPT = "a cat astronaut floating in space, watercolor style"
MODEL = "stabilityai/sdxl-turbo"


def sync():
    if use_gpu:
        torch.cuda.synchronize()


def run(dtype, tag, runs=3):
    """加载 -> 预热 -> 计时 -> 保存 -> 释放,返回 (平均耗时, 峰值显存GB)"""
    if use_gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    kwargs = {"torch_dtype": dtype, "use_safetensors": True}
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"  # 只有 fp16 才有对应权重文件,省下载和加载时间

    pipe = AutoPipelineForText2Image.from_pretrained(MODEL, **kwargs).to(device)
    pipe.set_progress_bar_config(disable=True)

    gen_kwargs = dict(
        prompt=PROMPT,
        num_inference_steps=1,
        guidance_scale=0.0,  # turbo 不需要 CFG,设 0 会跳过 negative 分支,速度翻倍
        height=512,
        width=512,  # sdxl-turbo 原生分辨率
    )

    with torch.inference_mode():
        # 预热:排除 kernel 加载 / MIOpen 调优的一次性开销
        pipe(**gen_kwargs, generator=torch.Generator(device).manual_seed(0))
        sync()

        t0 = time.perf_counter()
        for i in range(runs):
            image = pipe(
                **gen_kwargs, generator=torch.Generator(device).manual_seed(0)
            ).images[0]
        sync()
        avg = (time.perf_counter() - t0) / runs

    image.save(f"./data/cat_{tag}.png")
    peak = torch.cuda.max_memory_allocated() / 1024**3 if use_gpu else 0.0

    # 彻底释放,避免影响下一轮对比
    del pipe, image
    gc.collect()
    if use_gpu:
        torch.cuda.empty_cache()
    return avg, peak


if use_gpu:
    configs = [
        (torch.float16, "fp16"),
        (torch.bfloat16, "bf16"),  # RDNA4 原生支持 bf16,数值范围比 fp16 稳
        (torch.float32, "fp32"),
    ]
else:
    configs = [(torch.float32, "fp32")]

for dtype, tag in configs:
    avg, peak = run(dtype, tag)
    print(f"[{tag}] 平均耗时: {avg * 1000:.0f} ms/张 | 峰值显存: {peak:.2f} GB")
