import ast
import operator as op
from pathlib import Path

WORKSPACE = Path("./workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)

CONFIG = {"confirm_writes": True}  # 写文件前是否需要人工确认
MAX_READ_CHARS = 4000


# ==================== 计算器:绝不用 eval ====================
# eval("__import__('os').system('...')") 能执行任意代码,不能把模型的输出直接交给 eval。
# 这里用 ast 解析语法树,只放行四则运算和乘方。
_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def calculator(expression: str):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 100:  # 防止 9**9**9 卡死
                raise ValueError("指数过大")
            return _OPS[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("只支持数字和 + - * / // % ** 运算")

    return ev(ast.parse(expression, mode="eval").body)


# ==================== 文件工具:限制在 workspace 内 ====================
def safe_path(name: str) -> Path:
    p = (WORKSPACE / name).resolve()
    if not p.is_relative_to(WORKSPACE):  # 挡住 ../../etc/passwd 这类路径穿越
        raise ValueError("路径越界:只允许访问 workspace 目录内的文件")
    return p


def list_files():
    files = [str(p.relative_to(WORKSPACE)) for p in WORKSPACE.rglob("*") if p.is_file()]
    return "\n".join(sorted(files)) or "(workspace 是空的)"


def read_file(path: str):
    text = safe_path(path).read_text(encoding="utf-8")
    if len(text) > MAX_READ_CHARS:  # 工具输出太长会撑爆上下文,也更贵
        return text[:MAX_READ_CHARS] + f"\n...[已截断,文件共 {len(text)} 字符]"
    return text


def write_file(path: str, content: str):
    p = safe_path(path)
    if CONFIG["confirm_writes"]:
        print(f"\n⚠️  Agent 想写入文件:{path}({len(content)} 字符)")
        if input("允许吗?(y/n) ").strip().lower() != "y":
            return "用户拒绝了这次写入操作"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {path}"


def search_files(keyword: str):
    hits = []
    for p in WORKSPACE.rglob("*"):
        if p.is_file() and p.suffix in (".txt", ".md", ".csv"):
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if keyword in line:
                    hits.append(f"{p.relative_to(WORKSPACE)}:{i}: {line.strip()}")
    return "\n".join(hits[:20]) or f"没有找到包含「{keyword}」的内容"


# ==================== 注册表 + 给模型看的说明书 ====================
REGISTRY = {
    "calculator": calculator,
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "search_files": search_files,
}

TOOLS = [
    {
        "name": "calculator",
        "description": "计算数学表达式,支持 + - * / // % ** 和括号。任何涉及数字的计算都应使用它,不要心算。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "例如 (120*1500+80*2200)/2",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "list_files",
        "description": "列出 workspace 中所有文件。不确定文件名时先调用它。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": "读取 workspace 中某个文本文件的内容。过长的文件会被截断。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对 workspace 的路径,例如 sales.csv",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "创建或覆盖 workspace 中的文本文件。会覆盖同名文件,使用前确认路径正确。",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "search_files",
        "description": "在 workspace 所有 .txt/.md/.csv 文件中搜索包含某个关键词的行,返回文件名、行号和内容。",
        "input_schema": {
            "type": "object",
            "properties": {"keyword": {"type": "string"}},
            "required": ["keyword"],
        },
    },
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
]
