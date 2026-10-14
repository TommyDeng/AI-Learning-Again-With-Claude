import requests

r = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "texts": ["房间干净,服务态度很好,下次还来", "隔音太差,一晚上没睡好,再也不住了"]
    },
)
print(r.status_code)
print(r.json())

# 故意发非法输入,看服务如何拒绝
print(
    requests.post("http://127.0.0.1:8000/predict", json={"texts": []}).status_code
)  # 422
print(
    requests.post("http://127.0.0.1:8000/predict", json={"texts": ["   "]}).status_code
)  # 422
