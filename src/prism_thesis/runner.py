"""批量运行：(条件 × 题目 × 重复) 的每次运行写一行 JSON 到结果文件，支持中断后续跑。

- 每个工作线程使用自己的 MySQL 服务（Docker 模式下端口为 base_port + 线程号），互不干扰；
- 已完成的运行（按 run_key 识别）在续跑时跳过；基础设施错误（接口或数据库故障）不算完成，续跑时重做；
- 每条记录带实验指纹：模型与解码参数、智能体设置、提示版本、数据文件、随机种子、运行方式（付费 / 不付费）、
  与实验行为有关的全部代码（除命令行、输入检查与统计分析之外的所有模块），以及该条件使用的风险表内容、
  记忆库内容与检索设置、基线参数。续跑、合并结果与正式分析都要求指纹一致；任何一项改变（包括代码改动）都须换新的输出路径；
- 付费调用必须显式允许（allow_paid），并受费用上限 budget_usd 约束，超出即停止派发新运行。
  每次模型响应都记账：服务商返回 token 数时按其计费，缺失时按本地分词器计数估计（记录 cost_complete=False）。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import queue
import random
import subprocess
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

from . import __version__
from .agent import run_episode, task_features
from .config import ROOT, make_policy, memory_path, memory_settings, split_condition, table_key, table_path
from .data import Task
from .env import MySQLServer, TaskSession
from .tokens import TokenCounter


class ExperimentMismatch(RuntimeError):
    """结果文件中已有记录与当前实验配置不一致。"""


class CorruptResults(ExperimentMismatch):
    """结果文件不是完整的 JSONL（末行写入中断，或中间某行损坏）。"""


def run_key(condition: str, task_id: int, rep: int) -> str:
    return f'{condition}|{task_id}|{rep}'


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode('utf-8')


# 不影响运行行为的模块：改动它们不改变指纹。记忆库构建（memory/build.py）只影响记忆库文件，
# 而记忆库内容本身计入使用记忆的条件的指纹；这样修改构建代码不会使已完成的无记忆运行失效。
NON_EXPERIMENT_MODULES = frozenset({'cli.py', 'checks.py', 'stats.py', '__main__.py', 'memory/build.py'})


def experiment_modules() -> list[Path]:
    """与实验行为有关的代码文件（提示、解析、环境、判分、特征、策略、风险表、智能体循环、运行器等）。"""
    pkg = Path(__file__).resolve().parent
    return [f for f in sorted(pkg.rglob('*.py')) if f.relative_to(pkg).as_posix() not in NON_EXPERIMENT_MODULES]


def experiment_code_hash() -> str:
    pkg = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in experiment_modules():
        h.update(f.relative_to(pkg).as_posix().encode() + b'\0' + f.read_bytes() + b'\0')
    return h.hexdigest()[:16]


def experiment_fingerprint(cfg: dict, condition: str, paid: bool = True) -> str:
    """影响该条件运行结果的全部设置的哈希（不含线程数、费用上限、数据库地址等运行参数）。"""
    parts = {
        'model': cfg['model'], 'agent': cfg['agent'], 'prompt_variant': cfg.get('prompt_variant', 'standard'),
        'code': experiment_code_hash(), 'mode': 'paid' if paid else 'unpaid', 'mysql_image': cfg['mysql']['image'],
        'run_seed': cfg['run']['seed'],
        'data': {f: _sha((ROOT / 'data' / f).read_bytes()) for f in ('db_bench.parquet', 'splits.json')},
    }
    key = table_key(condition)
    if key:
        path = table_path(cfg, key)
        parts['risk_table'] = _sha(path.read_bytes()) if path.exists() else 'missing'
        parts['step_pp'] = cfg['prism']['step_pp']
    settings = memory_settings(cfg, condition)
    if settings is not None:            # 记忆库内容与检索设置（无记忆条件不依赖记忆库）
        path = memory_path(cfg)
        parts['memory'] = {'store': _sha(path.read_bytes()) if path.exists() else 'missing', 'retrieval': settings}
    if split_condition(condition)[0].partition('@')[0] in ('random_drop', 'llmlingua2'):
        parts['baselines'] = cfg['baselines']
    return _sha(_canon(parts))[:16]


def run_seed(base: int, key: str) -> int:
    return int(hashlib.sha256(f'{base}|{key}'.encode()).hexdigest()[:8], 16)


def read_lines(path: Path) -> list[dict]:
    """逐行读取结果。末行写入中断（不完整的 JSON 或缺少换行）与中间行损坏都报错；
    末行中断可用同一命令续跑修复（见 repair_tail）。"""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        raise CorruptResults(f'{path} 末行写入不完整（可能是运行中断）；请用同一命令续跑，续跑会先修复文件末尾')
    out = []
    for i, line in enumerate(data.decode('utf-8').split('\n'), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise CorruptResults(f'{path} 第 {i} 行不是完整的 JSON（{e.msg}），文件已损坏，需人工检查') from None
    return out


def repair_tail(path: Path, log: Callable[[str], None] = print) -> None:
    """续跑前修复中断写入留下的末行：完整的 JSON 只补换行；不完整的移到旁边的 .truncated-<时间> 文件，
    该次运行视为未完成、稍后重跑（其费用未记录）。中间行损坏不在此修复，由 read_lines 报错。"""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b'\n'):
        return
    cut = data.rfind(b'\n') + 1
    tail = data[cut:]
    try:
        json.loads(tail.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        stamp = time.strftime('%Y%m%dT%H%M%S')
        for n in itertools.count():      # 排他创建，不覆盖已有的备份
            side = path.with_name(f"{path.name}.truncated-{stamp}" + (f'-{n}' if n else ''))
            try:
                fh = side.open('xb')
            except FileExistsError:
                continue
            break
        with fh:
            fh.write(tail)
            fh.flush()
            os.fsync(fh.fileno())
        with path.open('r+b') as fh:     # 备份写好后才截断主文件
            fh.truncate(cut)
        log(f'警告：{path.name} 末行写入不完整（{len(tail)} 字节），已移到 {side.name}；该次运行将重跑，其费用未计入')
    else:
        with path.open('ab') as fh:
            fh.write(b'\n')
        log(f'警告：{path.name} 末行缺少换行，已补上')


def check_resume(existing: list[dict], fingerprints: dict[str, str]) -> None:
    """结果文件中凡是本次要运行的条件，其已有记录的指纹都必须与当前一致。"""
    bad = sorted({r['condition'] for r in existing if r['condition'] in fingerprints
                  and r.get('fingerprint') != fingerprints[r['condition']]})
    if bad:
        raise ExperimentMismatch(f'结果文件中条件 {bad} 的已有记录来自不同的实验（模型、提示、数据、代码、风险表或'
                                 '运行方式不同），不能续跑；请使用新的输出路径')


def completed_keys(path: Path) -> set[str]:
    return {r['run_key'] for r in read_lines(path) if r.get('status') != 'infrastructure_error'}


def load_records(paths: Iterable[Path]) -> list[dict]:
    """读取结果；同一 run_key 以最后一条非基础设施错误的记录为准。
    同一 run_key 出现不同指纹（来自不同实验配置）时报错，不静默覆盖。"""
    out: dict[str, dict] = {}
    for p in paths:
        for r in read_lines(Path(p)):
            if r.get('status') == 'infrastructure_error':
                continue
            old = out.get(r['run_key'])
            if old is not None and old.get('fingerprint') != r.get('fingerprint'):
                raise ExperimentMismatch(f"{r['run_key']} 在输入文件中有来自不同实验配置的记录")
            out[r['run_key']] = r
    return list(out.values())


def code_version() -> str:
    try:
        rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        rev = ''
    return f'{__version__}+{rev}' if rev else __version__


class MeteredLLM:
    """每次模型响应后记账，使中途失败的运行也计入已发生的费用。服务商未返回输入或输出 token 数时，
    用本地分词器对本次消息与回复计数作为估计，并标记 complete=False。"""

    def __init__(self, llm, counter: TokenCounter):
        self.llm, self.counter, self.model = llm, counter, getattr(llm, 'model', None)
        self.prompt_tokens = self.completion_tokens = self.calls = 0
        self.complete = True

    def chat(self, messages):
        resp = self.llm.chat(messages)
        self.calls += 1
        if resp.prompt_tokens is None or resp.completion_tokens is None:
            self.complete = False
        self.prompt_tokens += (resp.prompt_tokens if resp.prompt_tokens is not None
                               else self.counter.count_messages(messages))
        self.completion_tokens += (resp.completion_tokens if resp.completion_tokens is not None
                                   else self.counter.count(resp.content))
        return resp


def exit_code(state: dict) -> int:
    """0：全部完成；1：有基础设施错误、工作线程失败或未完成；3：因费用上限停止。"""
    if state.get('errors') or state.get('infra'):
        return 1
    if state.get('unfinished'):
        return 3 if state.get('stopped') else 1
    return 0


def run_experiment(cfg: dict, tasks: list[Task], conditions: list[str], reps: int, out: Path,
                   llm_factory: Callable[[Task], object], paid: bool, workers: Optional[int] = None,
                   log: Callable[[str], None] = print) -> dict:
    """返回 {'cost','ran','success','infra','stopped','unfinished','errors'}；用 exit_code() 判断是否全部完成。
    费用上限针对整个结果文件（含以前各次运行已记录的费用）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    counter = TokenCounter()
    for c in conditions:                 # 在主线程先构造一次，风险表缺失等错误立即报告
        make_policy(c, cfg, counter, allow_rehearsal_memory=not paid)
    fingerprints = {c: experiment_fingerprint(cfg, c, paid) for c in conditions}
    repair_tail(out, log)
    existing = read_lines(out)
    check_resume(existing, fingerprints)
    done = {r['run_key'] for r in existing if r.get('status') != 'infrastructure_error'}
    jobs = [(c, t, r) for r in range(reps) for t in tasks for c in conditions
            if run_key(c, t.task_id, r) not in done]
    prior_cost = sum(r.get('cost_usd_est') or 0.0 for r in existing)
    budget = cfg['run']['budget_usd']
    state = {'cost': prior_cost, 'ran': 0, 'success': 0, 'infra': 0, 'stopped': False,
             'unfinished': len(jobs), 'errors': []}
    log(f'待运行 {len(jobs)} 次（已完成 {len(done)} 次，已记录费用 {prior_cost:.4f} 美元）→ {out}')
    if not jobs:
        return state
    if paid and prior_cost >= budget:
        state['stopped'] = True
        log(f'已记录费用达到上限 {budget} 美元，不派发新运行')
        return state
    mcfg = cfg['mysql']
    n_workers = workers or cfg['run']['workers']
    if not mcfg['docker']:
        n_workers = 1
    n_workers = max(1, min(n_workers, len(jobs)))
    price = cfg['price_usd_per_mtok']
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()
    start_lock = threading.Lock()   # MySQL 容器逐个初始化：同时初始化多个时，慢磁盘上可能都超时
    version = code_version()

    def cost_of(p_in: int, p_out: int) -> float:
        return (p_in * price['input'] + p_out * price['output']) / 1e6 if paid else 0.0

    def worker(idx: int) -> None:
        server = MySQLServer(image=mcfg['image'], host=mcfg['host'], port=mcfg['base_port'] + idx,
                             user=mcfg['user'], password=mcfg['password'], docker=mcfg['docker'])
        try:
            with start_lock:
                server.start(timeout=float(mcfg.get('start_timeout_s', 600)))
            policies: dict = {}
            while not state['stopped']:
                try:
                    cond, task, rep = q.get_nowait()
                except queue.Empty:
                    return
                key = run_key(cond, task.task_id, rep)
                seed = run_seed(cfg['run']['seed'], key)
                if cond not in policies:
                    policies[cond] = make_policy(cond, cfg, counter, allow_rehearsal_memory=not paid)
                rec = {'run_key': key, 'condition': cond, 'split': task.split, 'task_id': task.task_id,
                       'group_id': task.group_id, 'rep': rep, 'seed': seed, 'code_version': version,
                       'fingerprint': fingerprints[cond],
                       'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                llm = None
                try:
                    llm = MeteredLLM(llm_factory(task), counter)
                    rec['model'] = llm.model
                    with TaskSession(server, task, agent_user=f'agent{idx}',
                                     statement_timeout_ms=cfg['agent']['statement_timeout_ms']) as session:
                        ep = run_episode(task, policies[cond], llm, session, counter,
                                         max_rounds=cfg['agent']['max_rounds'],
                                         first_action_only=cfg['agent']['first_action_only'],
                                         rng=random.Random(seed))
                    f = task_features(task)
                    rec.update(asdict(ep))
                    rec.update({'n_experiences': len({u for t in ep.turns for u in t['experiences']}),
                                'task_type_pred': f.task_type, 'is_query': task.is_query,
                                'predicted_functions': sorted(f.predicted_functions), 'n_turns': len(ep.turns)})
                    rec.update({'model_calls': llm.calls, 'cost_complete': llm.complete,
                                'cost_usd_est': cost_of(llm.prompt_tokens, llm.completion_tokens)})
                except Exception as e:  # noqa: BLE001  接口或数据库故障：记录后续跑时重做
                    rec.update({'status': 'infrastructure_error', 'error': repr(e),
                                'traceback': traceback.format_exc()[-2000:]})
                    if llm is not None:     # 已返回的模型响应照常计费
                        rec.update({'model_calls': llm.calls, 'cost_complete': llm.complete,
                                    'cost_usd_est': cost_of(llm.prompt_tokens, llm.completion_tokens)})
                with lock:
                    with out.open('a', encoding='utf-8') as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
                    if rec['status'] == 'infrastructure_error':
                        state['infra'] += 1
                    else:
                        state['ran'] += 1
                        state['success'] += int(rec['success'])
                    state['cost'] += rec.get('cost_usd_est') or 0.0
                    if paid and state['cost'] >= budget and not state['stopped']:
                        state['stopped'] = True
                        log(f'达到费用上限 {budget} 美元（含以前已记录的费用），停止派发新运行')
                    if (state['ran'] + state['infra']) % 50 == 0:
                        log(f"进度 {state['ran'] + state['infra']}/{len(jobs)}，成功 {state['success']}，"
                            f"基础设施错误 {state['infra']}，累计估计费用 {state['cost']:.4f} 美元")
        except Exception as e:  # noqa: BLE001  启动失败或线程内其他错误：交给主线程汇报
            with lock:
                state['errors'].append(f'工作线程 {idx}：{e!r}')
            log(f'工作线程 {idx} 失败：{e!r}')
        finally:
            server.stop()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    state['unfinished'] = len(jobs) - state['ran'] - state['infra']
    log(f"完成 {state['ran']} 次，成功 {state['success']}，基础设施错误 {state['infra']}，"
        f"未完成 {state['unfinished']} 次，累计估计费用 {state['cost']:.4f} 美元"
        + (f"；工作线程错误 {len(state['errors'])} 个" if state['errors'] else ''))
    return state
