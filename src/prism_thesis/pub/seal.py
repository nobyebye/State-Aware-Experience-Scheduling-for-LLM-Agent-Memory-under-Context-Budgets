"""测试集封存：封存划分（Environment.sealed_splits）上的付费运行必须满足

1. 给出预先登记文件（--prereg），该文件已提交到 git 且没有未提交的改动；
2. 实验代码、配置与数据（src/、configs/、data/）没有未提交的改动，使记录中的提交号能复现这次运行；
3. 同一环境、同一模型的测试集此前没有在别的输出路径上运行过；确需重跑时必须写明理由（--rerun-reason），
   理由与两次运行都留在使用记录里，论文中如实报告。
每次放行都在使用记录（docs/pub/test_usage.jsonl）追加一行：时间、环境、模型、条件、题数、重复次数、
输出路径、预先登记文件及其 SHA-256、提交号。使用记录随代码一起提交。
不付费的脚本化演练不读取模型输出，不受此限制，也不记录。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence

CLEAN_PATHS = ('src', 'configs', 'data')


class SealError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, timeout=30)


def prereg_info(prereg: Optional[str], root: Path) -> dict:
    if not prereg:
        raise SealError('封存的测试集只能在预先登记之后运行：请用 --prereg 给出已提交的预先登记文件')
    path = Path(prereg)
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise SealError(f'预先登记文件不存在：{path}')
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    if _git(root, 'ls-files', '--error-unmatch', rel).returncode != 0:
        raise SealError(f'预先登记文件 {rel} 尚未提交到 git')
    if _git(root, 'status', '--porcelain', '--', rel).stdout.strip():
        raise SealError(f'预先登记文件 {rel} 有未提交的改动')
    dirty = _git(root, 'status', '--porcelain', '--', *CLEAN_PATHS).stdout.strip()
    if dirty:
        raise SealError('代码、配置或数据有未提交的改动，测试集运行必须能由提交号复现：\n' + dirty[:500])
    head = _git(root, 'rev-parse', 'HEAD').stdout.strip()
    if not head:
        raise SealError('无法读取 git 提交号')
    return {'prereg': rel, 'prereg_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'commit': head}


def read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(x) for x in log_path.read_text(encoding='utf-8').splitlines() if x.strip()]


def authorize(log_path: Path, root: Path, env: str, split: str, model: str, conditions: Sequence[str],
              n_tasks: int, reps: int, out: str, prereg: Optional[str],
              rerun_reason: Optional[str] = None, now: Optional[str] = None) -> dict:
    """检查并登记一次封存划分的付费运行；不满足条件时抛 SealError。返回写入使用记录的一行。"""
    info = prereg_info(prereg, root)
    prior = [e for e in read_log(log_path) if (e['env'], e['split'], e['model']) == (env, split, model)]
    resumed = any(e['out'] == out for e in prior)
    others = sorted({e['out'] for e in prior if e['out'] != out})
    if others and not resumed and not rerun_reason:
        raise SealError(f'{env} 的 {split} 划分已经在模型 {model} 上运行过（{others}）。测试集只用一次；'
                        '确需重跑请用 --rerun-reason 写明理由，理由会留在使用记录中')
    if resumed and any(e['prereg_sha256'] != info['prereg_sha256'] for e in prior if e['out'] == out):
        raise SealError('续跑使用的预先登记文件与首次运行不同')
    entry = {'time': now or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'env': env, 'split': split,
             'model': model, 'conditions': list(conditions), 'n_tasks': n_tasks, 'reps': reps, 'out': out,
             **info, 'resumed': resumed, 'rerun_reason': rerun_reason}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return entry
