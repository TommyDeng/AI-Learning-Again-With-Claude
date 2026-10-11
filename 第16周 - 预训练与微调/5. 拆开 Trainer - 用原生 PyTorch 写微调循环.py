import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


def manual_finetune(model, train_ds, tok, epochs=2, lr=2e-5, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = train_ds.remove_columns(
        [
            c
            for c in train_ds.column_names
            if c not in ("input_ids", "attention_mask", "label")
        ]
    )
    train_ds = train_ds.rename_column(
        "label", "labels"
    )  # 模型的 forward 接收的参数名是 labels
    collator = DataCollatorWithPadding(tok)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps
    )

    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)  # 传入 labels 时,模型会自动算好交叉熵 loss
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )  # 梯度裁剪,防止梯度爆炸
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 20 == 0:
                print(
                    f"epoch {epoch + 1} step {step}/{len(loader)} loss={out.loss.item():.4f}"
                )
    return model


# 对照一下:zero_grad → forward → loss → backward → step 一个都没少。
# 多出来的只有学习率预热/衰减调度器和梯度裁剪,都是第11周见过的训练技巧。
