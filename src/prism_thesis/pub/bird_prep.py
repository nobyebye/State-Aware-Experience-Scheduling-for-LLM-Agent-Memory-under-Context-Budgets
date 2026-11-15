"""由 BIRD dev 的原始文件生成 BIRD-Agent 的数据（data/bird/）。只读取原始文件，不调用模型。

输入：dev.json（问题、提示、参考 SQL、难度）与 dev_databases/<db>/（<db>.sqlite 与 database_description/*.csv）。
步骤：
1. 在 SQLite 上执行每道题的参考 SQL，记录结果行集合的哈希。参考 SQL 出错、超时、结果为空或行数过多的题目
   排除（空结果的题目无法区分“答对”与“没查到”）；
2. 题目组：同一数据库内，参考 SQL 骨架（去掉字面量）相同或问题文本相同的题目并为一组，同组题目不跨划分；
3. 划分：按数据库分层，把题目组随机（固定种子）分到 build / search / calib / test，多出的进 reserve；
   build 用于构建记忆库，search 用于候选搜索，calib 用于非劣校准，test 为封存的一次性测试集；
4. 每个数据库的表信息：建表语句、列说明（来自 database_description）、每张表前 3 行样例；
5. 数据库文件清单（大小与 SHA-256）。
全部步骤是确定性的：相同输入与种子得到相同输出。
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from ..envs.bird import connect_readonly, result_hash, run_with_timeout

DEFAULT_SIZES = {'build': 350, 'search': 250, 'calib': 400, 'test': 500}
_STR = re.compile(r"'(?:[^']|'')*'")
_NUM = re.compile(r'\b\d+(?:\.\d+)?\b')


def skeleton(sql: str) -> str:
    s = _STR.sub('?', sql)
    s = _NUM.sub('0', s)
    return ' '.join(s.lower().replace(';', ' ').split())


def norm_question(q: str) -> str:
    return ' '.join(re.findall(r'[a-z0-9]+', q.lower()))


def gold_results(questions: Sequence[Mapping], db_root: Path, timeout_s: float = 30,
                 max_rows: int = 100000, log=print) -> list[dict]:
    """每道题加上 gold_hash / gold_rows / excluded / reason。"""
    out, conns = [], {}
    for i, q in enumerate(questions):
        db = q['db_id']
        if db not in conns:
            conns[db] = connect_readonly(Path(db_root) / db / f'{db}.sqlite')
        r = {k: q.get(k, '') for k in ('question_id', 'db_id', 'question', 'evidence', 'SQL', 'difficulty')}
        reason = None
        try:
            rows, overflow = run_with_timeout(conns[db], q['SQL'], timeout_s, max_rows)
            if overflow:
                reason = 'gold_too_large'
            elif not rows:
                reason = 'gold_empty'
            else:
                r['gold_hash'], r['gold_rows'] = result_hash(rows), len({tuple(x) for x in rows})
        except TimeoutError:
            reason = 'gold_timeout'
        except Exception as e:  # noqa: BLE001
            reason = f'gold_error: {str(e)[:80]}'
        r['excluded'], r['reason'] = reason is not None, reason
        out.append(r)
        if (i + 1) % 200 == 0:
            log(f'参考 SQL 已执行 {i + 1}/{len(questions)}')
    for c in conns.values():
        c.close()
    return out


def make_groups(tasks: Sequence[Mapping]) -> dict[int, str]:
    """题号 → 题目组 id（'<db>/g<编号>'，编号按组内最小题号排序）。"""
    parent = {t['question_id']: t['question_id'] for t in tasks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen: dict[tuple, int] = {}
    for t in tasks:
        for key in ((t['db_id'], 's', skeleton(t['SQL'])), (t['db_id'], 'q', norm_question(t['question']))):
            if key in seen:
                a, b = find(seen[key]), find(t['question_id'])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = t['question_id']
    db_of = {t['question_id']: t['db_id'] for t in tasks}
    roots = sorted({find(x) for x in parent})
    counter: dict[str, int] = defaultdict(int)
    name = {}
    for r in roots:
        name[r] = f'{db_of[r]}/g{counter[db_of[r]]:04d}'
        counter[db_of[r]] += 1
    return {x: name[find(x)] for x in parent}


def make_splits(tasks: Sequence[Mapping], groups: Mapping[int, str], sizes: Mapping[str, int] = DEFAULT_SIZES,
                seed: int = 20261115) -> dict[int, str]:
    """按数据库分层，把未排除题目的题目组分到各划分。可用题目少于各划分之和时按比例缩小。"""
    usable = [t for t in tasks if not t['excluded']]
    if not usable:
        return {}
    total = sum(sizes.values())
    scale = min(1.0, len(usable) / total) if total else 0.0
    by_db: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for t in usable:
        by_db[t['db_id']][groups[t['question_id']]].append(t['question_id'])
    out: dict[int, str] = {}
    order = list(sizes)
    for db in sorted(by_db):
        n_db = sum(len(v) for v in by_db[db].values())
        quota = {s: sizes[s] * scale * n_db / len(usable) for s in order}
        have = {s: 0 for s in order}
        gs = sorted(by_db[db], key=lambda g: (-len(by_db[db][g]),
                                              hashlib.sha256(f'{seed}|{g}'.encode()).hexdigest()))
        for g in gs:
            best = max(order, key=lambda s: (quota[s] - have[s], -order.index(s)))
            split = best if quota[best] - have[best] >= 0.5 else 'reserve'
            for tid in by_db[db][g]:
                out[tid] = split
            if split != 'reserve':
                have[split] += len(by_db[db][g])
    return out


def _read_csv(path: Path) -> list[dict]:
    raw = path.read_bytes()
    for enc in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return list(csv.DictReader(io.StringIO(text)))


def _clean(s, n: int = 200) -> str:
    s = ' '.join(str(s or '').split())
    return s if len(s) <= n else s[:n] + '…'


def column_descriptions(desc_dir: Path) -> dict[str, dict[str, str]]:
    """表名（小写）→ 列名（小写）→ 说明（列说明与取值说明合并；与列名相同的说明略去）。"""
    out: dict[str, dict[str, str]] = {}
    if not desc_dir.is_dir():
        return out
    for f in sorted(desc_dir.glob('*.csv')):
        cols = {}
        for row in _read_csv(f):
            row = {(k or '').strip().lower(): v for k, v in row.items()}
            name = (row.get('original_column_name') or '').strip()
            d = _clean(row.get('column_description'))
            v = _clean(row.get('value_description'))
            if d.lower().replace('_', ' ') == name.lower().replace('_', ' '):
                d = ''
            text = '; '.join(x for x in (d, v) if x)
            if name and text:
                cols[name.lower()] = _clean(text, 300)
        out[f.stem.lower()] = cols
    return out


def db_schema(db_id: str, db_root: Path, sample_rows: int = 3) -> dict:
    conn = connect_readonly(Path(db_root) / db_id / f'{db_id}.sqlite')
    descs = column_descriptions(Path(db_root) / db_id / 'database_description')
    tables = []
    try:
        names = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                             'ORDER BY name').fetchall()
        for name, ddl in names:
            info = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
            d = descs.get(name.lower(), {})
            cols = [{'name': c[1], 'type': c[2], 'description': d.get(c[1].strip().lower(), '')} for c in info]
            samples = [list(r) for r in conn.execute(f'SELECT * FROM "{name}" LIMIT {int(sample_rows)}').fetchall()]
            samples = [[v if isinstance(v, (int, float, str)) or v is None else '<blob>' for v in r] for r in samples]
            ddl = '\n'.join(' '.join(line.split()) for line in (ddl or '').splitlines() if line.strip())
            tables.append({'name': name, 'ddl': ddl, 'columns': cols, 'samples': samples})
    finally:
        conn.close()
    return {'db_id': db_id, 'tables': tables}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(dev_json: Path, db_root: Path, out_dir: Path, sizes: Mapping[str, int] = DEFAULT_SIZES,
            seed: int = 20261115, timeout_s: float = 30, log=print) -> dict:
    questions = json.loads(Path(dev_json).read_text(encoding='utf-8'))
    for i, q in enumerate(questions):
        q.setdefault('question_id', i)
    tasks = gold_results(questions, db_root, timeout_s, log=log)
    groups = make_groups(tasks)
    splits = make_splits(tasks, groups, sizes, seed)
    out_dir = Path(out_dir)
    (out_dir / 'schema').mkdir(parents=True, exist_ok=True)
    with (out_dir / 'tasks.jsonl').open('w', encoding='utf-8') as fh:
        for t in tasks:
            fh.write(json.dumps({k: v for k, v in t.items() if k not in ('excluded', 'reason')},
                                ensure_ascii=False) + '\n')
    meta = [{'task_id': t['question_id'], 'split': splits.get(t['question_id'], 'excluded'),
             'group_id': groups[t['question_id']], 'excluded': t['excluded'], 'reason': t['reason']} for t in tasks]
    (out_dir / 'splits.json').write_text(json.dumps(
        {'seed': seed, 'sizes': dict(sizes), 'tasks': meta}, ensure_ascii=False, indent=1), encoding='utf-8')
    dbs = sorted({t['db_id'] for t in tasks})
    manifest = {}
    for db in dbs:
        (out_dir / 'schema' / f'{db}.json').write_text(
            json.dumps(db_schema(db, db_root), ensure_ascii=False, indent=1), encoding='utf-8')
        p = Path(db_root) / db / f'{db}.sqlite'
        manifest[db] = {'bytes': p.stat().st_size, 'sha256': file_sha256(p)}
    (out_dir / 'db_manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    counts: dict[str, int] = defaultdict(int)
    for m in meta:
        counts[m['split']] += 1
    reasons: dict[str, int] = defaultdict(int)
    for t in tasks:
        if t['excluded']:
            reasons[t['reason'].split(':')[0]] += 1
    summary = {'questions': len(tasks), 'databases': len(dbs), 'splits': dict(counts), 'excluded': dict(reasons),
               'groups': len(set(groups.values()))}
    log(json.dumps(summary, ensure_ascii=False))
    return summary
