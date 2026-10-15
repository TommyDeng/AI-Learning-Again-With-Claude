# 模拟一个场景:两组人的真实能力分布完全相同,但历史录用记录对 B 组有偏见(同样能力,B 组更难被录用)。
# 现实中你拿不到"真实是否合格",这里是模拟数据才有,所以能看清模型到底学到了什么。

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

pd.set_option("display.float_format", lambda v: f"{v:.3f}")
pd.set_option("display.width", 200)
rng = np.random.default_rng(0)
N = 30000

group = rng.integers(0, 2, N)  # 0 = A 组, 1 = B 组
skill = rng.normal(0, 1, N)  # 真实能力:两组分布完全相同
proxy = 1.0 * group + rng.normal(
    0, 0.7, N
)  # 代理特征:与组别强相关(如毕业学校类型、居住区域)
qualified = (skill > 0).astype(int)  # 真实是否合格(现实中拿不到)

# 历史标签带偏见:同样能力,B 组更难被录用
logit = 1.8 * skill - 1.0 * group
hired = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

df = pd.DataFrame(
    dict(group=group, skill=skill, proxy=proxy, qualified=qualified, hired=hired)
)
print(df.groupby("group")[["qualified", "hired"]].mean())

train, rest = train_test_split(df, test_size=0.5, random_state=0)
val, test = train_test_split(rest, test_size=0.5, random_state=1)


def report(name, d, pred):
    out = []
    for g, gname in [(0, "A"), (1, "B")]:
        m = (d["group"] == g).values
        q = d["qualified"].values[m] == 1
        p = pred[m]
        out.append(
            dict(
                方案=name,
                组别=gname,
                选中率=p.mean(),
                合格者被选中率=p[q].mean(),
                不合格者被选中率=p[~q].mean(),
            )
        )
    return pd.DataFrame(out)


def fit_scores(cols, train_df, eval_df, weights=None):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_df[cols], train_df["hired"], sample_weight=weights)
    return clf.predict_proba(eval_df[cols])[:, 1], clf


def reweighing_weights(d):
    """重加权(Kamiran & Calders):让'组别'和'标签'在加权后的训练集里相互独立"""
    g, y = d["group"].values, d["hired"].values
    w = np.zeros(len(d))
    for gv in (0, 1):
        for yv in (0, 1):
            m = (g == gv) & (y == yv)
            w[m] = (g == gv).mean() * (y == yv).mean() / m.mean()
    return w


def equalize_selection(scores, groups, target_rate):
    """后处理:为每个组单独定阈值,使两组选中率都等于 target_rate"""
    pred = np.zeros(len(scores), dtype=int)
    for gv in (0, 1):
        m = groups == gv
        thr = np.quantile(scores[m], 1 - target_rate)
        pred[m] = (scores[m] >= thr).astype(int)
    return pred


tables = []

# 方案1:直接用所有特征,包括组别
s1, _ = fit_scores(["skill", "proxy", "group"], train, test)
tables.append(report("1 含组别特征", test, (s1 >= 0.5).astype(int)))

# 方案2:"眼不见为净":把组别特征删掉
s2, clf2 = fit_scores(["skill", "proxy"], train, test)
tables.append(report("2 去掉组别特征", test, (s2 >= 0.5).astype(int)))

# 方案3:预处理,重加权
w = reweighing_weights(train)
s3, _ = fit_scores(["skill", "proxy"], train, test, weights=w)
tables.append(report("3 重加权", test, (s3 >= 0.5).astype(int)))

# 方案4:后处理,分组阈值(阈值在验证集上定,测试集只用来评估)
sv, _ = fit_scores(["skill", "proxy"], train, val)
target = (sv >= 0.5).mean()
tables.append(
    report("4 分组阈值", test, equalize_selection(s2, test["group"].values, target))
)

print(pd.concat(tables, ignore_index=True).to_string(index=False))


# pip install fairlearn

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from sklearn.metrics import recall_score

pred = (s2 >= 0.5).astype(int)
mf = MetricFrame(
    metrics={"选中率": selection_rate, "TPR": recall_score},
    y_true=test["qualified"],
    y_pred=pred,
    sensitive_features=test["group"],
)
print(mf.by_group)
print(
    "人口统计均等差:",
    demographic_parity_difference(
        test["qualified"], pred, sensitive_features=test["group"]
    ),
)
print(
    "均等几率差:",
    equalized_odds_difference(
        test["qualified"], pred, sensitive_features=test["group"]
    ),
)
