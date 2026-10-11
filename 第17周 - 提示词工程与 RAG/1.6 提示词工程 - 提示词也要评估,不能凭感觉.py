from llm import ask

test_set = [
    ("点击导出后页面一直转圈", "bug"),
    ("能不能加个批量删除功能", "功能建议"),
    ("请问发票怎么开", "咨询"),
    ("客服小姐姐态度超级好", "表扬"),
    ("上传图片总是失败,提示网络错误", "bug"),
    ("希望支持导出成 PDF", "功能建议"),
    ("会员到期后数据会保留多久?", "咨询"),
    ("新版界面很清爽,赞", "表扬"),
    ("手机端打开就闪退", "bug"),
    ("如果能和钉钉打通就好了", "功能建议"),
]

prompt_v1 = "把用户反馈归类为 bug / 功能建议 / 咨询 / 表扬,只输出类别。\n\n反馈:{x}"

prompt_v2 = """你是客服工单分类员。把用户反馈归为以下四类之一,只输出类别名:
- bug:现有功能出错、崩溃、异常
- 功能建议:希望新增或改进功能(包括"如果能……就好了"这类含蓄表达)
- 咨询:询问规则、价格、用法
- 表扬:称赞产品或服务

反馈:{x}"""


def evaluate(template):
    correct, errors = 0, []
    for text, label in test_set:
        pred = ask(template.format(x=text), temperature=0).strip()
        if pred == label:
            correct += 1
        else:
            errors.append((text, label, pred))
    return correct / len(test_set), errors


for name, tpl in [("v1 简单版", prompt_v1), ("v2 带定义", prompt_v2)]:
    acc, errors = evaluate(tpl)
    print(f"{name}: 准确率 {acc:.0%}")
    for e in errors:
        print("   错误:", e)
