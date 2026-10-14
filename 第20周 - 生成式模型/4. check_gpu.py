import torch

print("torch:", torch.__version__)  # 应该带 +rocm
print("HIP:", torch.version.hip)  # 不是 None 就说明是 ROCm 版
print("可用:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

x = torch.randn(4000, 4000, device="cuda")
print("矩阵乘法测试:", (x @ x).sum().item())


# 方法三:在 VS Code 里运行

# 你之前报错的路径看起来是在编辑器里运行的,可以让它直接用新环境:

# 按 Ctrl+Shift+P,输入 Python: Select Interpreter
# 选择 C:\venvs\rocm\Scripts\python.exe(没有的话选"Enter interpreter path"手动输入)
# 之后点右上角的运行按钮,就会用新环境了
