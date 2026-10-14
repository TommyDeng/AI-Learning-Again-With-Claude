import json
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "./models/chn-sentiment-final")
LOG_PATH = os.getenv("LOG_PATH", "./models/predictions.jsonl")
MAX_LEN = 128  # 送进模型的最大 token 数
MAX_BATCH = 32  # 单次请求最多多少条文本
MAX_CHARS = 2000  # 单条文本最长字符数,防止恶意超长输入拖垮服务
LABELS = {0: "负面", 1: "正面"}

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 【阶段一：服务启动时执行】
    # 在这里加载一次模型和分词器，而不是每次收到请求都重新加载（那样每次推理都要几秒钟）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, local_files_only=True
        )
        .to(device)
        .eval()
    )
    card_path = os.path.join(MODEL_DIR, "model_card.json")
    card = (
        json.load(open(card_path, encoding="utf-8"))
        if os.path.exists(card_path)
        else {}
    )

    # 将加载好的模型、分词器和配置存入全局的 state 字典中，方便后续的 API 接口调用
    state.update(tok=tok, model=model, device=device, card=card)

    yield  # ─── 华丽分割线：服务正式开始接收并处理客户端请求 ─── 启动和关闭逻辑写在同一个函数里（yield 上方是启动，下方是关闭）

    # 【阶段二：服务关闭时执行】
    # 清理全局状态，释放显存/内存
    state.clear()


app = FastAPI(title="中文情感分类服务", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)

    @field_validator("texts")
    @classmethod
    def check_texts(cls, v):
        for t in v:
            if not t.strip():
                raise ValueError("文本不能为空")
            if len(t) > MAX_CHARS:
                raise ValueError(f"单条文本不能超过 {MAX_CHARS} 字符")
        return v


class Prediction(BaseModel):
    label: str
    label_id: int
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_sha256: str
    latency_ms: float


@app.get("/health")
def health():
    """给负载均衡器、Docker、K8s 探活用:模型加载好了才算健康"""
    return {
        "status": "ok" if "model" in state else "loading",
        "device": state.get("device"),
    }


@app.get("/model-info")
def model_info():
    return state.get("card", {})


# 注意这里是 def 而不是 async def:模型推理是阻塞的 CPU/GPU 计算,
# 用 async def 会堵住整个事件循环,让其他请求全部排队。
# 普通 def 会被 FastAPI 放进线程池执行。
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    tok, model, device = state["tok"], state["model"], state["device"]

    batch = tok(
        req.texts,
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():  # 比 no_grad 更彻底,推理时的标准写法
        probs = torch.softmax(model(**batch).logits, dim=-1).cpu()
    conf, ids = probs.max(dim=-1)

    preds = [
        Prediction(label=LABELS[int(i)], label_id=int(i), confidence=round(float(c), 4))
        for i, c in zip(ids, conf)
    ]
    latency = (time.perf_counter() - t0) * 1000

    # 监控日志:记录元信息,不记录用户原文(隐私)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.writelines(
            json.dumps(
                {
                    "ts": time.time(),
                    "text_len": len(text),
                    "label_id": p.label_id,
                    "confidence": p.confidence,
                    "latency_ms": round(latency, 2),
                },
                ensure_ascii=False,
            )
            + "\n"
            for text, p in zip(req.texts, preds)
        )

    return PredictResponse(
        predictions=preds,
        model_sha256=state["card"].get("weights_sha256", "unknown")[:12],
        latency_ms=round(latency, 2),
    )


# uvicorn app:app --app-dir 'c:\AI-Learning-Again-With-Claude\第21周 - MLOps 与模型部署' --host 127.0.0.1 --port 8000
