"""经验条目与记忆库文件。

三类条目（kind）：
  example  成功示例：任务指令、执行成功的 SQL 与最终答案；
  repair   修复经验：错误签名、出错的 SQL 与改正后的 SQL（来自“出错后改正并成功”的运行）；
  insight  洞见：模型离线从成功与失败轨迹的对比中提炼的自然语言规则（参照 ExpeL）。
每条记录来源题目、题目组与划分，用于同组排除与防泄漏检查；token 数在运行时用分词器计算。

上下文中的经验单元 id 为 'exp:<槽位>:<条目 id>'，槽位由类型字母与检索排名组成（i 洞见、x 示例、r 修复），
渲染时按槽位排序，保证同一组单元在每轮的文本与顺序一致。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

KINDS = ('example', 'repair', 'insight')
# 经验的选择组：第 1 条相似示例、其余相似示例、洞见、修复经验
EXP_GROUPS = ('exp_top1', 'exp_rest', 'insight', 'repair')
SLOT_PREFIX = {'insight': 'i', 'example': 'x', 'repair': 'r'}
SLOT_ORDER = {'i': 0, 'x': 1, 'r': 2}
MEMORY_HEADER = 'Experience from past tasks (for reference only; the current task and table may differ):'
FORBIDDEN_SPLITS = frozenset({'test'})


@dataclass(frozen=True)
class Experience:
    eid: str
    kind: str                         # example / repair / insight
    text: str                         # 注入上下文的文本
    key_text: str                     # 检索用文本（示例与修复为来源任务指令，洞见为其本身）
    task_type: str                    # query / modify / any
    source_tasks: tuple[int, ...]
    source_groups: tuple[str, ...]
    source_splits: tuple[str, ...]
    error_key: Optional[str] = None   # 修复经验：MySQL 错误号或错误类别
    signature: Optional[str] = None   # 修复经验：错误号 + 出错 SQL 的结构特征


@dataclass
class MemoryStore:
    entries: list[Experience] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.by_id = {e.eid: e for e in self.entries}
        if len(self.by_id) != len(self.entries):
            raise ValueError('记忆库中有重复的条目 id')
        bad = sorted(e.eid for e in self.entries if set(e.source_splits) & FORBIDDEN_SPLITS)
        if bad:
            raise ValueError(f'记忆库含来自测试集的条目 {bad[:5]}…，测试集题目不得进入记忆库')
        unknown = sorted({e.kind for e in self.entries} - set(KINDS))
        if unknown:
            raise ValueError(f'未知的经验类型 {unknown}')

    def of_kind(self, kind: str) -> list[Experience]:
        return [e for e in self.entries if e.kind == kind]

    def counts(self) -> dict[str, int]:
        return {k: len(self.of_kind(k)) for k in KINDS}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {'meta': self.meta, 'entries': [asdict(e) for e in self.entries]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'MemoryStore':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        entries = []
        for e in d['entries']:
            e = dict(e)
            for k in ('source_tasks', 'source_groups', 'source_splits'):
                e[k] = tuple(e[k])
            entries.append(Experience(**e))
        return cls(entries, d.get('meta', {}))

    def content_hash(self) -> str:
        data = json.dumps([asdict(e) for e in self.entries], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]


def make_uid(kind: str, rank: int, eid: str) -> str:
    return f'exp:{SLOT_PREFIX[kind]}{rank:02d}:{eid}'


def is_exp(uid: str) -> bool:
    return uid.startswith('exp:')


def eid_of(uid: str) -> str:
    return uid.split(':', 2)[2]


def _slot_key(uid: str):
    slot = uid.split(':', 2)[1]
    return SLOT_ORDER[slot[0]], int(slot[1:]), uid


def ordered(uids: Iterable[str]) -> list[str]:
    return sorted((u for u in uids if is_exp(u)), key=_slot_key)


def exp_text(uid: str, store: MemoryStore) -> str:
    return store.by_id[eid_of(uid)].text


def render_memory(uids: Iterable[str], store: Optional[MemoryStore]) -> str:
    """把经验单元渲染为一个经验块（标题 + 各条目，空行分隔）；没有经验单元时为空串。"""
    ids = ordered(uids)
    if not ids:
        return ''
    if store is None:
        raise ValueError('计划中有经验单元，但没有提供记忆库')
    return '\n\n'.join([MEMORY_HEADER] + [exp_text(u, store) for u in ids])
