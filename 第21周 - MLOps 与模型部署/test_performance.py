import concurrent.futures as cf
import statistics
import time

import requests

URL = "http://127.0.0.1:8000/predict"
payload = {"texts": ["这家酒店的位置不错,房间也很干净"] * 4}


def one_call(_):
    t0 = time.perf_counter()
    r = requests.post(URL, json=payload, timeout=30)
    return (time.perf_counter() - t0) * 1000, r.status_code


for concurrency in [1, 4, 16]:
    n = 60
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(concurrency) as ex:
        results = list(ex.map(one_call, range(n)))
    wall = time.perf_counter() - t0
    lat = sorted(r[0] for r in results)
    ok = sum(r[1] == 200 for r in results)
    print(
        f"并发 {concurrency:2d} | 成功 {ok}/{n} | 吞吐 {n / wall:5.1f} 请求/秒 | "
        f"p50 {statistics.median(lat):6.0f}ms | p95 {lat[int(n * 0.95) - 1]:6.0f}ms"
    )


# 看延迟分位数(p50/p95/p99)而不是平均值,因为用户体验往往被最慢的那几个请求决定。并发提高后,吞吐不再增长而延迟飙升,说明你碰到了瓶颈。

# 常见的优化手段:

# 批处理: 一次处理多条比逐条处理便宜得多,GPU 上尤其明显
# 更小的模型: 蒸馏、量化(INT8)能明显提速
# 多进程: uvicorn app:app --workers 4,每个 worker 各加载一份模型,内存占用会相应翻倍
# 专用推理服务: 大模型场景下有专门的推理引擎(如 vLLM),这里先了解有这类东西即可
