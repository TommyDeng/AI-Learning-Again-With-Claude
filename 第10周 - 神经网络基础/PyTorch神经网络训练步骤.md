## PyTorch 核心训练循环（标准 5 步法）

在 PyTorch 中，神经网络的每一轮（Epoch）训练都遵循一个非常固定且规范的执行流程：


### 1. 梯度清零
清除上一次计算留下的梯度，防止梯度累加。
```python
optimizer.zero_grad()
```

### 2. 前向传播（预测）
数据前向传播，算出模型的当前预测结果。
```python
outputs = model(X_train_t)
```

### 3. 计算损失（评估）
计算预测值与真实值之间的差距（Loss）。
```python
loss = criterion(outputs, y_train_t)
```

### 4. 反向传播（算梯度）
基于链式法则自动计算各参数的梯度 $\frac{\partial \text{Loss}}{\partial w}$。
```python
loss.backward()
```

### 5. 参数更新（优化）
根据计算出的梯度和学习率（Learning Rate）调整参数，让 Loss 变小。
```python
optimizer.step()
```


### 详细步骤拆解与原理说明

1. **`optimizer.zero_grad()` —— 梯度归一（清零）**
    
    - **作用**：将模型所有可学习参数的梯度重置为 $0$。
        
    - **原理**：PyTorch 在默认情况下**梯度是累加的**（`grad += new_grad`）。为了避免上一轮训练的梯度影响当前轮次，每轮训练前必须显式进行清零。
        
2. **`model(X_train_t)` —— 前向传播（预测）**
    
    - **作用**：把输入数据传给模型，计算出当前的预测输出值（`outputs`）。
        
    - **原理**：数据按照网络定义的 `forward` 函数顺序流动，经过各层的加权求和与非线性激活函数，得到最终的预测结果。
        
3. **`criterion(outputs, y_train_t)` —— 评估损失（求 Loss）**
    
    - **作用**：对比模型的预测值与真实的标签，计算出一个衡量误差的标量值（`loss`）。
        
    - **原理**：损失函数（如 `CrossEntropyLoss` 或 `MSELoss`）量化了模型“当前预测得有多差”，Loss 越小说明模型性能越好。
        
4. **`loss.backward()` —— 反向传播（计算梯度）**
    
    - **作用**：自动计算损失函数关于模型各个权重和偏置的梯度。
        
    - **原理**：利用**链式法则（Chain Rule）**，从最后的损失值开始沿着网络反向层层传递，通过 PyTorch 的自动微分引擎（Autograd）求出每个参数对 Loss 的偏导数 $\frac{\partial \text{Loss}}{\partial w}$。
        
5. **`optimizer.step()` —— 参数更新（梯度下降）**
    
    - **作用**：根据上一步算好的梯度，更新模型的所有权重和偏置参数。
        
    - **原理**：优化器（如 SGD、Adam）沿着梯度的反方向（使 Loss 减少最快的方向）按指定的学习率 $lr$ 调整参数，公式如：
        
        $$w \leftarrow w - lr \cdot \frac{\partial \text{Loss}}{\partial w}$$
        

> **💡 记忆顺口溜**：**一清零，二预测，三算损失，四传梯度，五更新。**