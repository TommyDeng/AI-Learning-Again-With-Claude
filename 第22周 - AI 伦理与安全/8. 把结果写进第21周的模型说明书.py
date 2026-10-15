import json

card_path = "./models/chn-sentiment-final/model_card.json"
card = json.load(open(card_path, encoding="utf-8"))

card["intended_use"] = "中文酒店/餐饮评论的情感倾向分类,用于汇总分析"
card["out_of_scope"] = [
    "对个人做决策(如评价员工)",
    "非评论类文本",
    "讽刺、反语较多的场景",
]
card["fairness_eval"] = {
    "method": "身份词反事实替换",
    "max_prob_gap": 0.03,
    "date": "2026-10",
}  # 换成你实测的数字
card["known_limitations"] = ["对否定和转折句错误率更高", "训练数据只覆盖酒店领域"]
card["last_reviewed"] = "2026-10"

json.dump(card, open(card_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
