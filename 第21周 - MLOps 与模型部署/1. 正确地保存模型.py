import hashlib
import json
import time
from pathlib import Path


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_model_card(model_dir, metrics, base_model, data_desc, notes=""):
    """给模型目录写一份说明书:这个模型是什么、怎么来的、效果如何、文件有没有被改过"""
    d = Path(model_dir)
    weights = next(d.glob("model.safetensors"), None) or next(
        d.glob("pytorch_model.bin")
    )
    card = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": base_model,
        "training_data": data_desc,
        "metrics": metrics,
        "weights_file": weights.name,
        "weights_sha256": sha256_of(weights),
        "notes": notes,
    }
    (d / "model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return card


if __name__ == "__main__":
    card = write_model_card(
        "./models/chn-sentiment-final",
        metrics={"accuracy": 0.93},  # 换成你第16周实际测得的数字
        base_model="hfl/chinese-roberta-wwm-ext",
        data_desc="ChnSentiCorp 酒店评论,训练 2000 条",
    )
    print(json.dumps(card, ensure_ascii=False, indent=2))
