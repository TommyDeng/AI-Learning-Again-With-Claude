import torch
from diffusers import AutoPipelineForText2Image

device = "cuda" if torch.cuda.is_available() else "cpu"
use_gpu = device == "cuda"
print("使用设备:", torch.cuda.get_device_name(0) if use_gpu else "CPU")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16 if use_gpu else torch.float32,
    variant="fp16" if use_gpu else None,
).to(device)

image = pipe(
    "a cat astronaut floating in space, watercolor style",
    num_inference_steps=1,
    guidance_scale=0.0,
).images[0]
image.save("./data/cat16.png")


# fp32

device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float32,  # 改成 fp32
    # 去掉 variant="fp16",直接加载完整精度的权重
).to(device)

image = pipe(
    "a cat astronaut floating in space, watercolor style",
    num_inference_steps=1,
    guidance_scale=0.0,
).images[0]
image.save("./data/cat32.png")

if device == "cuda":
    print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
