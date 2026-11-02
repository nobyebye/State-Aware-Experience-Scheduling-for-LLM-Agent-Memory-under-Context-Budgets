"""分析脚本共用的路径、数据与计数工具（零调用）。

所有路径都相对本文件定位，不依赖当前工作目录。旧仓库（E:\\硕士毕业论文）只读：
仅从中读取运行器系统提示、SQL 函数列表与 first_action，不在其中写入任何内容。
旧仓库位置由 --legacy-repo 或环境变量 PRISM_LEGACY_REPO 指定。
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
DEFAULT_DATA = ANALYSIS_DIR / 'n_runs_extract.jsonl'
DEFAULT_TOKENIZER = PROJECT_ROOT / 'data' / 'tokenizer' / 'llama31_8b_tokenizer.json'
LEGACY_FILES = {
    'validation': Path('src/agent_memory/benchmark/validation.py'),
    'runner': Path('src/agent_memory/runner/__init__.py'),
    'actions': Path('src/agent_memory/actions.py'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser(description, legacy=True):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA, help='抽取的基础运行（JSONL）')
    ap.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER, help='Llama-3.1 tokenizer.json')
    if legacy:
        ap.add_argument('--legacy-repo', type=Path, default=os.environ.get('PRISM_LEGACY_REPO'),
                        help='旧仓库根目录（只读）；也可用环境变量 PRISM_LEGACY_REPO')
    return ap


def _assigned_value(path, name):
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == name for t in node.targets):
            return node.value
    raise SystemExit(f'{path} 中找不到 {name}')


def load_legacy(repo):
    """返回 (运行器指令部分 sp, SQL 函数集合, first_action)。"""
    if repo is None:
        raise SystemExit('请用 --legacy-repo 或环境变量 PRISM_LEGACY_REPO 指定旧仓库根目录')
    repo = Path(repo).resolve()
    paths = {k: repo / v for k, v in LEGACY_FILES.items()}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit('旧仓库缺少文件: ' + ', '.join(missing))
    call = _assigned_value(paths['validation'], 'SQL_FUNCTIONS')  # frozenset({...})
    funcs = ast.literal_eval(call.args[0])
    value = _assigned_value(paths['runner'], 'SYSTEM_PROMPT')  # """...""" + 函数列表
    while isinstance(value, ast.BinOp):
        value = value.left
    sp = value.value
    spec = importlib.util.spec_from_file_location('legacy_actions', paths['actions'])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return sp, funcs, mod.first_action, {k: sha256(p) for k, p in paths.items()}


def make_counter(tokenizer_path):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(tokenizer_path))
    tk.no_truncation(); tk.no_padding()
    return lambda s: len(tk.encode(s or '', add_special_tokens=False).ids)


def load_rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def function_list(funcs):
    return '\nAvailable SQL functions: ' + ', '.join(sorted(funcs)) + '.\n'


class Context:
    """第一步与第四步脚本的共同输入。"""

    def __init__(self, args):
        for p in (args.data, args.tokenizer):
            if not Path(p).is_file():
                raise SystemExit(f'找不到文件: {p}')
        self.sp, self.FUNCS, self.first_action, legacy_hash = load_legacy(args.legacy_repo)
        self.SYSTEM = self.sp + function_list(self.FUNCS)
        self.c = make_counter(args.tokenizer)
        self.rows = load_rows(args.data)
        print(f'# data={args.data} sha256={sha256(args.data)[:16]} tokenizer sha256={sha256(args.tokenizer)[:16]} '
              f'legacy={ {k: v[:16] for k, v in legacy_hash.items()} }', file=sys.stderr)
        print('system total', self.c(self.SYSTEM), '| function list', self.c(function_list(self.FUNCS)),
              '| instructions', self.c(self.sp))


def context(description):
    return Context(parser(description).parse_args())
