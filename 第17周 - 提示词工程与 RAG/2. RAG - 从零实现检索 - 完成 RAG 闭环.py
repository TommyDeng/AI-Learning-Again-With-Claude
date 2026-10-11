import numpy as np
from kb import DOCS
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# ---------- 分块 ----------
def chunk_text(text, size=200, overlap=40):
    """按字符切块,相邻块有重叠,避免一句话被切断后丢失上下文"""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


chunks = []  # 每个元素: {"id", "source", "text"}
for title, content in DOCS.items():
    for i, c in enumerate(chunk_text(content)):
        chunks.append({"id": f"{title}-{i}", "source": title, "text": c})
print("共", len(chunks), "个块")

# ---------- 建索引 ----------
vectors = encoder.encode([c["text"] for c in chunks], normalize_embeddings=True)
print("向量矩阵形状:", vectors.shape)  # 归一化后,点积 = 余弦相似度


# ---------- 检索 ----------
def retrieve(query, k=3):
    q = encoder.encode([query], normalize_embeddings=True)[0]
    scores = vectors @ q  # 第3周的矩阵乘法:一次算出与所有块的相似度
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]


for q in [
    "我想请年假要怎么申请?",
    "出差住酒店能报多少钱",
    "能不能把代码贴给 ChatGPT 帮我改",
]:
    print(f"\n问题:{q}")
    for c, s in retrieve(q):
        print(f"  {s:.3f} [{c['source']}] {c['text'][:30]}...")


from llm import ask


def build_prompt(question, retrieved):
    docs = "\n".join(
        f'<document id="{i + 1}" source="{c["source"]}">\n{c["text"]}\n</document>'
        for i, (c, _) in enumerate(retrieved)
    )
    return f"""你是公司内部的 HR 问答助手。请只根据下面提供的文档回答问题。

规则:
- 只使用文档中的信息,不要使用你自己的常识来补充具体数字或规定
- 如果文档中没有足够信息回答,直接说"根据现有资料无法回答这个问题",不要猜测
- 回答末尾用 [来源:文档编号] 的形式标注依据

<documents>
{docs}
</documents>

问题:{question}"""


def rag_answer(question, k=3, verbose=False):
    retrieved = retrieve(question, k)
    if verbose:
        for c, s in retrieved:
            print(f"   检索到 [{c['source']}] 相似度 {s:.3f}")
    return ask(build_prompt(question, retrieved), temperature=0)


questions = [
    "试用期的员工能远程办公吗?",
    "我去上海出差,住宿最多能报多少?",
    "年假没休完会怎么样?",
    "公司有没有提供健身房会员补贴?",  # 知识库里没有这个信息
]
for q in questions:
    print(f"\n问:{q}")
    print("答:", rag_answer(q, verbose=True))

# 重点看最后一个问题。 知识库里没有健身房补贴的信息,一个"没有约束"的模型很可能编出一个听起来合理的答案。
# 提示词里"无法回答就明说"这条规则,正是 RAG 抑制幻觉的关键。你可以把这条规则删掉,对比一下效果。

# 2. 换成向量数据库(ChromaDB)
# 文档量大了以后,每次全量矩阵乘法太慢,而且向量要持久化保存。向量数据库提供:存储、近似最近邻索引、元数据过滤
import chromadb

chroma = chromadb.PersistentClient(path=".data/chroma_db")
col = chroma.get_or_create_collection("handbook", metadata={"hnsw:space": "cosine"})

col.upsert(  # upsert:重复运行不会因 id 冲突报错
    ids=[c["id"] for c in chunks],
    documents=[c["text"] for c in chunks],
    embeddings=vectors.tolist(),
    metadatas=[{"source": c["source"]} for c in chunks],
)
print("库内条数:", col.count())


def retrieve_chroma(query, k=3, source=None):
    q = encoder.encode([query], normalize_embeddings=True)
    res = col.query(
        query_embeddings=q.tolist(),
        n_results=k,
        where={"source": source} if source else None,  # 元数据过滤
    )
    out = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(
            ({"source": meta["source"], "text": doc}, 1 - dist)
        )  # cosine 距离 = 1 - 相似度
    return out


for c, s in retrieve_chroma("加班能换休息吗"):
    print(f"{s:.3f} [{c['source']}] {c['text'][:30]}...")


# 最后一条可以直接加一个检索阈值:
def rag_answer_safe(question, k=3, threshold=0.35):
    retrieved = retrieve(question, k)
    if retrieved[0][1] < threshold:  # 最相关的块也不够相关,直接拒答
        return "知识库中没有找到相关内容。"
    return ask(build_prompt(question, retrieved), temperature=0)


# 混合检索:加入关键词匹配:(可以和上面的检索阈值一起使用)
# pip install rank_bm25 jieba
import jieba
from rank_bm25 import BM25Okapi

tokenized = [list(jieba.cut(c["text"])) for c in chunks]
bm25 = BM25Okapi(tokenized)


def hybrid_retrieve(query, k=3, alpha=0.5):
    # 向量分数
    q = encoder.encode([query], normalize_embeddings=True)[0]
    vec_scores = vectors @ q
    # BM25 分数,归一化到 0~1 再加权
    bm = np.array(bm25.get_scores(list(jieba.cut(query))))
    bm = bm / bm.max() if bm.max() > 0 else bm
    final = alpha * vec_scores + (1 - alpha) * bm
    top = np.argsort(-final)[:k]
    return [(chunks[i], float(final[i])) for i in top]


# alpha 控制两种信号的权重。有大量产品型号、人名、条款编号的场景,关键词那一路尤其有用。
