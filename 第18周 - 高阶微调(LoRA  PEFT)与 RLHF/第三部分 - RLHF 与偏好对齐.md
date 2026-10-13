## 第三部分:RLHF 与偏好对齐

### 1. 问题:预训练模型只会"续写"

第13周你训练的迷你 GPT 和第15周的 GPT-2,目标都是**预测下一个词**。问它"怎么煮咖啡?",它可能续写成"怎么煮茶?怎么煮奶茶?",因为网上这种问题列表很常见。它没有"听懂指令并有帮助地回答"的概念。

从"续写机器"变成"助手",通常分三步:

```
1. 预训练     海量文本上预测下一个词              → 学会语言和世界知识
2. SFT        (指令, 理想回答) 的示范数据微调      → 学会"按指令回答"的格式和风格
3. 偏好对齐   人类对多个回答的好坏排序            → 学会"哪种回答更好"
```

**你刚做的 JSON 提取微调,本质上就是第 2 步 SFT 的缩小版。**

### 2. 为什么 SFT 还不够

SFT 只能教模型模仿示范。但"什么是好回答"往往很难写出标准答案,却**很容易比较**:给人看两个回答,他能说出哪个更好(更准确、更安全、更清晰)。偏好对齐就是利用这种比较信号。

### 3. 经典 RLHF 流程

```
① 对同一个提示,让模型生成多个回答
② 人类标注员排序:回答A 好于 回答B
③ 训练一个"奖励模型":给任意回答打分,使得好回答分更高
④ 用强化学习(如 PPO)优化语言模型,最大化奖励模型的分数,
   同时加 KL 惩罚,防止模型为了刷分偏离原模型太远
```

#### 奖励模型的训练目标(Bradley-Terry)

python

```python
import torch
import torch.nn.functional as F

def reward_model_loss(r_chosen, r_rejected):
    """让 chosen 的分数高于 rejected:-log σ(r_chosen - r_rejected)"""
    return -F.logsigmoid(r_chosen - r_rejected).mean()

# 奖励模型打分已经"排对了"的情形
print(reward_model_loss(torch.tensor([2.0, 1.5]), torch.tensor([0.5, -1.0])))   # loss 小
# 打分"排反了"的情形
print(reward_model_loss(torch.tensor([0.5, -1.0]), torch.tensor([2.0, 1.5])))   # loss 大
```

只看分差,不关心绝对分数,这和你第7周学的二分类交叉熵是同一个形式。

#### 为什么要 KL 惩罚

如果只最大化奖励,模型会钻奖励模型的空子,生成一些奖励模型喜欢、但人类实际上觉得很怪的东西(**奖励黑客**,reward hacking)。KL 惩罚让优化后的模型不要离原来的模型太远。

### 4. DPO:跳过奖励模型的简化方案

PPO 流程复杂(要同时维护策略模型、参考模型、奖励模型、价值模型),训练不稳定。**DPO(直接偏好优化)** 证明了:可以直接用偏好对训练语言模型,不需要显式的奖励模型和强化学习循环。

python

```python
import torch
import torch.nn.functional as F

def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """
    pi_*  : 当前模型对 回答 的对数概率之和(越大表示模型越倾向这个回答)
    ref_* : 冻结的参考模型(通常是 SFT 后的模型)对同样回答的对数概率
    """
    chosen_gain = pi_chosen - ref_chosen          # 相比参考模型,更倾向 chosen 了多少
    rejected_gain = pi_rejected - ref_rejected    # 相比参考模型,更倾向 rejected 了多少
    logits = beta * (chosen_gain - rejected_gain)
    return -F.logsigmoid(logits).mean()

ref_c, ref_r = torch.tensor([-20.0]), torch.tensor([-20.0])      # 参考模型对两个回答一视同仁

# 情况1:模型还没学到偏好(与参考模型一样)
print("未训练:", dpo_loss(torch.tensor([-20.0]), torch.tensor([-20.0]), ref_c, ref_r))
# 情况2:模型已经更倾向 chosen、更排斥 rejected
print("学到偏好:", dpo_loss(torch.tensor([-15.0]), torch.tensor([-25.0]), ref_c, ref_r))
# 情况3:模型学反了
print("学反了:", dpo_loss(torch.tensor([-25.0]), torch.tensor([-15.0]), ref_c, ref_r))
```

**直觉理解:** loss 在"提高 chosen 的概率、降低 rejected 的概率"的方向上下降,而且是**相对于参考模型**来衡量的,所以天然带有"别偏离参考模型太远"的约束(`beta` 控制这个约束的强度)。

#### 如何计算序列的对数概率

DPO 里的 `pi_chosen` 怎么来?就是第13周迷你 GPT 里 logits 的用法:

python

```python
def sequence_logprob(model, input_ids, attention_mask, labels):
    """一条回答的对数概率 = 回答部分每个 token 的 log p 之和"""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    targets = labels[:, 1:]
    mask = targets != -100                                    # 只统计回答部分,提示部分不算
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1)
```

训练 DPO 时,每个 batch 要对 chosen 和 rejected 各算一次,当前模型算一次、冻结的参考模型再算一次。实际工程里一般用 Hugging Face 的 `trl` 库(`DPOTrainer`),**它的接口版本之间变化较大,我这里不贴具体调用,以官方文档为准**。理解了上面这个 loss,看 `trl` 的源码和文档都会很顺。

### 5. 更新的方向(了解即可)

- **基于可验证奖励的强化学习:** 对数学题、代码这类有标准答案的任务,可以直接用"答案对不对、测试过不过"作为奖励,不需要人类标注。这是近年推理类模型进步的重要来源。**GRPO** 是其中常见的算法,去掉了价值模型,更省资源。
- **RLAIF / 宪法式方法:** 用 AI 依据一套明确原则来生成偏好标注,降低对人类标注的依赖。

这几个方向变化很快,具体细节需要时去读最新的论文和文档。

---

## 本周项目

**核心项目:** 把上面的 LoRA 脚本跑通,并做三组对比实验,记录结果。

python

```python
# 在 main() 里把训练部分包装成函数 train_lora(r, n_train, target_modules) 后:
experiments = [
    dict(r=2,  n_train=300, target=["q_proj", "v_proj"]),
    dict(r=8,  n_train=300, target=["q_proj", "k_proj", "v_proj", "o_proj"]),
    dict(r=8,  n_train=50,  target=["q_proj", "k_proj", "v_proj", "o_proj"]),
    dict(r=16, n_train=300, target=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]),
]
```

每组记录:**可训练参数量、训练耗时、JSON 合法率、字段正确率**,整理成表格。回答这几个问题:

1. 数据量从 300 降到 50,效果掉多少?(这是 LoRA 对小数据友好程度的直接体现)
2. 只加 `q_proj/v_proj` 和加满注意力四个投影,差别大吗?
3. 加上 MLP 层、`r=16`,是不是值得多花的参数和时间?

**进阶练习:** 把任务换成你自己关心的格式,比如"把一句口语化的需求转成固定字段的指令"。改 `make_example` 生成你自己的数据,保持"可自动评分"这个特点,否则你无法客观判断微调有没有用。