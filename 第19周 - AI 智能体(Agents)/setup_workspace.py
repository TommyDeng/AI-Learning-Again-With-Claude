from pathlib import Path

ws = Path("./workspace")
(ws / "notes").mkdir(parents=True, exist_ok=True)

(ws / "sales.csv").write_text(
    "product,units,price\n耳机,120,1500\n键盘,80,2200\n鼠标,300,600\n显示器,45,8800\n",
    encoding="utf-8",
)

(ws / "notes" / "meeting.txt").write_text(
    "周会纪要\n本周主要进展:完成了用户模块的重构。\n风险:测试环境不稳定。\n"
    "待办:下周三前补齐接口文档。\n",
    encoding="utf-8",
)

(ws / "notes" / "ideas.md").write_text(
    "# 想法\n- 给产品加一个深色模式\n- 研究如何用 RAG 搭建内部问答\n", encoding="utf-8"
)

print("workspace 已准备好")
