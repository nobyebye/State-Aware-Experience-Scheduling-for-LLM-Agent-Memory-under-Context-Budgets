"""发表线命令行：python -m prism_thesis.pub <命令> --config configs/pub/<环境>.json ...

  check-data    各划分的题目数与题目组数，划分之间题目组是否重叠（BIRD 加 --hash-dbs 核对数据库文件）
  verify-env    在真实环境中执行每道题的参考解法并判分（不调用模型）
  show-prompt   打印某道题在某个条件、某个状态下发送的消息与 token 数
  dry-run       用脚本化智能体跑完整流程（不调用模型、不产生费用）
  run           正式运行（调用模型，必须加 --allow-paid；封存的测试集还必须给出 --prereg）
  build-memory  由 build 划分上 nomem 条件的运行记录构建经验记忆库（不调用模型）
硕士论文的命令行（python -m prism_thesis）与 db_bench 环境不受影响。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from ..agent import build_messages
from ..config import ROOT, RehearsalArtifact, load_config, make_policy, memory_path
from ..env import ExecResult
from ..envs import get_env
from ..memory import MemoryStore
from ..memory import build as mbuild
from ..policies.base import TurnState
from ..runner import ExperimentMismatch, exit_code, load_records, run_experiment
from ..tokens import TokenCounter
from . import seal


def _tasks(env, args):
    split = None if args.split == 'all' else args.split
    tasks = env.load_tasks(split)
    if getattr(args, 'task_ids', None):
        wanted = {int(x) for x in args.task_ids.split(',')}
        tasks = [t for t in tasks if t.task_id in wanted]
    if getattr(args, 'limit', None):
        tasks = tasks[:args.limit]
    return tasks


def cmd_check_data(args, cfg, env):
    tasks = env.load_tasks()
    by: dict[str, list] = {}
    for t in tasks:
        by.setdefault(t.split, []).append(t)
    groups = {s: {t.group_id for t in ts} for s, ts in by.items()}
    for s in sorted(by):
        print(f'{s:8s} 题目 {len(by[s]):5d}  题目组 {len(groups[s]):5d}')
    overlap = sorted((a, b) for a in groups for b in groups if a < b and groups[a] & groups[b])
    print('划分之间题目组无重叠：' + ('是' if not overlap else f'否 {overlap}'))
    excluded = len(env.load_tasks(include_excluded=True)) - len(tasks)
    print(f'排除的题目：{excluded}')
    ok = not overlap
    if env.name == 'bird':
        from .bird_prep import file_sha256
        manifest = json.loads((env.data_dir / 'db_manifest.json').read_text(encoding='utf-8'))
        for db, m in sorted(manifest.items()):
            p = env.db_path(db)
            good = p.exists() and p.stat().st_size == m['bytes'] and \
                (not args.hash_dbs or file_sha256(p) == m['sha256'])
            ok &= good
            if not good:
                print(f'数据库文件与清单不符或不存在：{p}')
        print(f"数据库文件 {len(manifest)} 个：{'与清单一致' if ok else '有问题'}"
              + ('' if args.hash_dbs else '（只核对了大小；加 --hash-dbs 核对 SHA-256）'))
    return 0 if ok else 1


def cmd_verify_env(args, cfg, env):
    tasks = _tasks(env, args)
    backend = env.make_backend(cfg, 'verify')
    backend.start()
    bad = []
    try:
        for t in tasks:
            llm = env.scripted('gold', t)
            history: list[dict] = []
            with env.open_session(backend, t, 'verify', cfg) as s:
                status = 'round_limit'
                for _ in range(cfg['agent']['max_rounds']):
                    p = env.parse(env.first_action(llm.chat(history).content))
                    history.append({'role': 'assistant', 'content': ''})
                    if p.kind != 'operation':
                        status = 'answered' if p.kind == 'answer' else 'invalid_action'
                        break
                    s.execute(p.content)
                ok, _ = env.grade(t, s, status, 'done')
            if not ok:
                bad.append(t.task_id)
    finally:
        backend.stop()
    print(f'参考解法判为成功：{len(tasks) - len(bad)}/{len(tasks)}' + (f'；失败题目 {bad[:50]}' if bad else ''))
    return 0 if not bad else 1


def cmd_show_prompt(args, cfg, env):
    counter = TokenCounter()
    task = next(t for t in env.load_tasks(include_excluded=True) if t.task_id == args.task_id)
    policy = make_policy(args.condition, cfg, counter, allow_rehearsal_memory=True)
    policy.start(task, env.task_features(task), random.Random(0))
    last = {'initial': None, 'after_success': ExecResult(True, '[]', None, 'ok'),
            'after_error': ExecResult(False, 'Error: example', None, 'other')}[args.phase]
    turn = 0 if last is None else 1
    plan = policy.plan(TurnState(turn, env.phase(turn, last), last, None))
    history = [] if last is None else [{'role': 'assistant', 'content': '(previous action)'},
                                       {'role': 'user', 'content': last.text}]
    messages, _ = build_messages(task, plan, history, policy.store, env)
    for msg in messages:
        print(f"--- {msg['role']} ({counter.count(msg['content'])} tokens)\n{msg['content']}")
    print(f'--- 本轮输入共 {counter.count_messages(messages)} tokens；省略的组：{plan.info.get("dropped", [])}')
    return 0


def _summary(recs):
    by: dict[str, list] = {}
    for r in recs:
        by.setdefault(r['condition'], []).append(r)
    for c, rs in sorted(by.items()):
        print(f"{c:28s} 运行 {len(rs):4d}  成功率 {sum(r['success'] for r in rs) / len(rs):.3f}  "
              f"平均输入 token {sum(r['input_tokens'] for r in rs) / len(rs):.1f}")


def cmd_dry_run(args, cfg, env):
    out = Path(args.out)
    if out.exists() and not args.resume:
        out.unlink()
    try:
        state = run_experiment(cfg, _tasks(env, args), args.conditions.split(','), args.reps, out,
                               lambda task: env.scripted(args.agent, task), paid=False, workers=args.workers)
    except (ExperimentMismatch, FileNotFoundError, RehearsalArtifact, ValueError) as e:
        print(f'拒绝运行：{e}')
        return 1
    if exit_code(state):
        print(f"演练未全部完成：基础设施错误 {state['infra']} 次，未完成 {state['unfinished']} 次，"
              f"工作线程错误 {state['errors']}")
        return exit_code(state)
    _summary(load_records([out]))
    return 0


def cmd_run(args, cfg, env):
    if not args.allow_paid:
        print('正式运行会调用付费接口。确认已获批准后加 --allow-paid 再运行。')
        return 2
    from .. import llm as llm_mod
    from ..cli import PatientLLM
    tasks = _tasks(env, args)
    conditions = args.conditions.split(',')
    m = cfg['model']
    if args.split in env.sealed_splits:
        if getattr(args, 'task_ids', None) or getattr(args, 'limit', None):
            print('拒绝运行：封存的测试集必须整体运行，不能用 --limit 或 --task-ids 取子集')
            return 1
        log_path = Path(cfg.get('pub', {}).get('test_usage_log', 'docs/pub/test_usage.jsonl'))
        log_path = log_path if log_path.is_absolute() else ROOT / log_path
        try:
            entry = seal.authorize(log_path, ROOT, env.name, args.split, m['model'], conditions, len(tasks),
                                   args.reps, str(args.out), args.prereg, args.rerun_reason)
        except seal.SealError as e:
            print(f'拒绝运行：{e}')
            return 1
        print(f"测试集运行已登记（{log_path.name}）：预先登记 {entry['prereg']}，提交 {entry['commit'][:10]}")
    llm = llm_mod.OpenAICompatibleLLM(m['model'], m['api_base'], m['api_key_env'], m['temperature'], m['top_p'],
                                      m['max_tokens'], m['timeout'], m.get('extra_body'))
    if args.rate_limit_retries > 0:
        llm = PatientLLM(llm, args.rate_limit_retries, args.rate_limit_wait)
    if args.budget_usd is not None:
        cfg['run']['budget_usd'] = args.budget_usd
    try:
        state = run_experiment(cfg, tasks, conditions, args.reps, Path(args.out), lambda task: llm, paid=True,
                               workers=args.workers)
    except (ExperimentMismatch, FileNotFoundError, RehearsalArtifact, ValueError) as e:
        print(f'拒绝运行：{e}')
        return 1
    code = exit_code(state)
    if code == 3:
        print(f"因费用上限停止，尚有 {state['unfinished']} 次未运行；提高上限后用相同命令续跑")
    elif code:
        print(f"运行未全部完成：基础设施错误 {state['infra']} 次，未完成 {state['unfinished']} 次，"
              f"工作线程错误 {state['errors']}；排除故障后用相同命令续跑")
    return code


def cmd_build_memory(args, cfg, env):
    from ..cli import _insight_file
    try:
        recs = load_records([Path(p) for p in args.runs])
    except ExperimentMismatch as e:
        print(f'拒绝构建：{e}')
        return 1
    recs = [r for r in recs if r['condition'] == 'nomem']
    outside = sorted({r['split'] for r in recs} - set(env.memory_splits))
    if outside:
        print(f'拒绝构建：记录中有 {outside} 划分的题目，记忆库只能由 {list(env.memory_splits)} 划分构建')
        return 1
    if not args.rehearsal:
        want = {t.task_id for t in env.load_tasks(env.memory_splits)}
        have = {r['task_id'] for r in recs}
        reps = cfg['reps']['memory_build']['nomem']
        short = [t for t in want if sum(r['task_id'] == t for r in recs) < reps]
        if have != want or short:
            print(f'输入检查未通过：nomem 覆盖 {len(have & want)}/{len(want)} 道题，'
                  f'{len(short)} 道题不足 {reps} 次重复')
            return 1
        if any(r.get('model', '').startswith('scripted') for r in recs):
            print('拒绝构建：记录来自脚本化智能体；演练请加 --rehearsal')
            return 1
    tasks = {t.task_id: t for t in env.load_tasks(include_excluded=True)}
    entries = mbuild.extract_examples(recs, tasks, cfg['memory'].get('max_answer_chars', 300), env) + \
        mbuild.extract_repairs(recs, tasks, env)
    try:
        if args.insights:
            entries += _insight_file(args.insights, args.rehearsal)
        store = MemoryStore(entries, meta={
            'env': env.name,
            'runs': [{'path': str(p), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in args.runs],
            'n_records': len(recs), 'insights': str(args.insights) if args.insights else None,
            'rehearsal': bool(args.rehearsal), 'splits': sorted({r['split'] for r in recs})})
    except ValueError as e:
        print(f'拒绝构建：{e}')
        return 1
    out = Path(args.out) if args.out else memory_path(cfg)
    store.save(out)
    c = store.counts()
    print(f"记忆库 {len(store.entries)} 条（成功示例 {c['example']}，修复经验 {c['repair']}，洞见 {c['insight']}）"
          f"，来自 {len(recs)} 次运行 → {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='prism_thesis.pub', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('check-data')
    p.add_argument('--hash-dbs', action='store_true')
    p = sub.add_parser('verify-env')
    p.add_argument('--split', default='all')
    p.add_argument('--limit', type=int)
    p.add_argument('--task-ids')
    p = sub.add_parser('show-prompt')
    p.add_argument('--task-id', type=int, required=True)
    p.add_argument('--condition', default='full')
    p.add_argument('--phase', default='initial', choices=('initial', 'after_success', 'after_error'))
    for name in ('dry-run', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--split', required=True)
        p.add_argument('--conditions', required=True)
        p.add_argument('--reps', type=int, default=1)
        p.add_argument('--out', required=True)
        p.add_argument('--workers', type=int)
        p.add_argument('--limit', type=int)
        p.add_argument('--task-ids')
        if name == 'dry-run':
            p.add_argument('--agent', default='gold', choices=('gold', 'gold_after_error'))
            p.add_argument('--resume', action='store_true')
        else:
            p.add_argument('--allow-paid', action='store_true')
            p.add_argument('--budget-usd', type=float)
            p.add_argument('--rate-limit-retries', type=int, default=30)
            p.add_argument('--rate-limit-wait', type=float, default=60)
            p.add_argument('--prereg', help='封存的测试集：已提交的预先登记文件')
            p.add_argument('--rerun-reason', help='封存的测试集：同一环境与模型再次运行的理由')
    p = sub.add_parser('build-memory')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--insights')
    p.add_argument('--out')
    p.add_argument('--rehearsal', action='store_true')
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    env = get_env(cfg)
    if env.name == 'db_bench':
        print('db_bench 环境请使用 python -m prism_thesis（硕士论文的命令行）')
        return 2
    return globals()['cmd_' + args.cmd.replace('-', '_')](args, cfg, env)


if __name__ == '__main__':
    sys.exit(main())
