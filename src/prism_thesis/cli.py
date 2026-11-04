"""命令行入口：python -m prism_thesis <命令> ...

  check-data      检查题目划分与数量（不需要 MySQL）
  verify-env      在 MySQL 中执行每道题的参考 SQL，检查判分器判为成功（不调用模型）
  show-prompt     打印某道题在某个条件下第一轮发送的消息
  dry-run         用脚本化智能体跑完整流程（不调用模型、不产生费用）
  run             正式运行（调用模型，必须加 --allow-paid）
  build-memory    由构建集与开发集上 nomem 条件的运行记录构建经验记忆库（不调用模型）
  distill-insights 离线提炼洞见（ExpeL 式，调用模型，必须加 --allow-paid；--rehearsal 用脚本化回复）
  show-memory     打印某道题检索到的经验（同组排除后）
  memory-stats    经验条目被发送的次数与所在运行的成功率（记忆演化的扩展分析）
  risk-estimate   由消融运行记录估计风险表
  calibrate       在校准集记录上做固定顺序非劣检验，确定 ε*
  compare         各条件相对完整提示的成功率差与 token 节省（组 bootstrap）
  token-audit     核对本地 token 计数与服务商返回的 prompt_tokens
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
import sys
import time
from pathlib import Path

from . import stats
from .checks import REHEARSAL_BANNER, check_records
from .agent import build_messages, task_features
from .config import ROOT, RehearsalArtifact, load_config, make_policy, make_retriever, memory_path, memory_settings
from .data import load_tasks
from .env import MySQLServer, TaskSession
from .evaluate import match_direct
from .features import phase, task_type
from .memory import MemoryStore
from .memory import build as mbuild
from .memory.store import Experience
from .policies.base import TurnState
from .prism.risk import estimate, single_only_records
from .runner import ExperimentMismatch, exit_code, load_records, run_experiment
from .tokens import TokenCounter

SPLITS = {'dev': 'development', 'development': 'development', 'construction': 'construction',
          'calib': 'construction', 'test': 'test', 'all': None, 'memory': ('construction', 'development')}
MEMORY_SPLITS = ('construction', 'development')
EXPECTED = {'development': (110, 67), 'construction': (272, 242), 'test': (114, 100)}


def _load_split(split):
    if isinstance(split, tuple):
        return sorted((t for s in split for t in load_tasks(s)), key=lambda t: t.task_id)
    return load_tasks(split)


def _tasks(args):
    tasks = _load_split(SPLITS[args.split])
    if getattr(args, 'task_ids', None):
        wanted = {int(x) for x in args.task_ids.split(',')}
        tasks = [t for t in tasks if t.task_id in wanted]
    if getattr(args, 'limit', None):
        tasks = tasks[:args.limit]
    return tasks


def _groups() -> dict[int, str]:
    return {t.task_id: t.group_id for t in load_tasks(include_excluded=True)}


def cmd_check_data(args, cfg):
    ok = True
    all_ids = set()
    for split, (n, g) in EXPECTED.items():
        ts = load_tasks(split)
        groups = {t.group_id for t in ts}
        all_ids |= groups
        good = (len(ts), len(groups)) == (n, g)
        ok &= good
        print(f'{split:13s} 题目 {len(ts):4d}  题目组 {len(groups):4d}  期望 {n}/{g}  {"通过" if good else "不符"}')
    per_split = {s: {t.group_id for t in load_tasks(s)} for s in EXPECTED}
    overlap = any(per_split[a] & per_split[b] for a in per_split for b in per_split if a < b)
    print('划分之间题目组无重叠：' + ('是' if not overlap else '否'))
    return 0 if ok and not overlap else 1


def cmd_verify_env(args, cfg):
    m = cfg['mysql']
    split = SPLITS[args.split]
    tasks = [t for t in load_tasks(include_excluded=True) if split is None or t.split in (split if isinstance(split, tuple) else (split,))]
    bad = []
    with MySQLServer(m['image'], m['host'], m['base_port'], m['user'], m['password'], m['docker']) as srv:
        for t in tasks:
            with TaskSession(srv, t) as s:
                r = s.execute(t.reference_sql)
                ok = r.ok and (s.table_md5() == t.answer_md5 if t.answer_md5 else match_direct(r.text, t.answer_direct))
            if not ok:
                bad.append(t.task_id)
    print(f'参考 SQL 判为成功：{len(tasks) - len(bad)}/{len(tasks)}' + (f'；失败题目 {bad}' if bad else ''))
    return 0 if not bad else 1


def cmd_show_prompt(args, cfg):
    counter = TokenCounter()
    task = next(t for t in load_tasks(include_excluded=True) if t.task_id == args.task_id)
    import random
    policy = make_policy(args.condition, cfg, counter, allow_rehearsal_memory=True)
    policy.start(task, task_features(task), random.Random(0))
    plan = policy.plan(TurnState(0, phase(0, None), None))
    messages, _ = build_messages(task, plan, [], policy.store)
    for msg in messages:
        print(f"--- {msg['role']} ({counter.count(msg['content'])} tokens)\n{msg['content']}")
    print(f'--- 本轮输入共 {counter.count_messages(messages)} tokens；单元：{sorted(plan.units)}')
    return 0


def cmd_dry_run(args, cfg):
    from . import scripted
    out = Path(args.out)
    if out.exists() and not args.resume:
        out.unlink()
    try:
        state = run_experiment(cfg, _tasks(args), args.conditions.split(','), args.reps, out,
                               lambda task: scripted.make(args.agent, task), paid=False, workers=args.workers)
    except (ExperimentMismatch, FileNotFoundError, RehearsalArtifact) as e:
        print(f'拒绝运行：{e}')
        return 1
    if exit_code(state):
        print(f"演练未全部完成：基础设施错误 {state['infra']} 次，未完成 {state['unfinished']} 次，"
              f"工作线程错误 {state['errors']}")
        return exit_code(state)
    recs = load_records([out])
    by = {}
    for r in recs:
        by.setdefault(r['condition'], []).append(r)
    for c, rs in sorted(by.items()):
        print(f"{c:28s} 运行 {len(rs):4d}  成功率 {sum(r['success'] for r in rs) / len(rs):.3f}  "
              f"平均输入 token {sum(r['input_tokens'] for r in rs) / len(rs):.1f}")
    return 0


def cmd_run(args, cfg):
    if not args.allow_paid:
        print('正式运行会调用付费接口。确认已获批准后加 --allow-paid 再运行。')
        return 2
    from .llm import OpenAICompatibleLLM
    m = cfg['model']
    llm = OpenAICompatibleLLM(m['model'], m['api_base'], m['api_key_env'], m['temperature'], m['top_p'],
                              m['max_tokens'], m['timeout'], m.get('extra_body'))
    if args.budget_usd is not None:
        cfg['run']['budget_usd'] = args.budget_usd
    try:
        state = run_experiment(cfg, _tasks(args), args.conditions.split(','), args.reps, Path(args.out),
                               lambda task: llm, paid=True, workers=args.workers)
    except (ExperimentMismatch, FileNotFoundError, RehearsalArtifact) as e:
        print(f'拒绝运行：{e}')
        return 1
    code = exit_code(state)
    if code == 3:
        print(f"因费用上限停止，尚有 {state['unfinished']} 次未运行；提高上限后用相同命令续跑")
    elif code:
        print(f"运行未全部完成：基础设施错误 {state['infra']} 次，未完成 {state['unfinished']} 次，"
              f"工作线程错误 {state['errors']}；排除故障后用相同命令续跑")
    return code


def cmd_risk_estimate(args, cfg):
    try:
        recs = load_records([Path(p) for p in args.runs])
    except ExperimentMismatch as e:
        print(f'拒绝估计：{e}')
        return 1
    pairs = cfg['complementary_pairs']
    if args.single_only:                 # 消融对照：完全不用互补机制，联合消融记录不参与也不要求
        recs, pairs = single_only_records(recs), []
    conds = sorted({r['condition'] for r in recs})
    joints = _required_joints(conds, pairs)
    required = ['full'] + sorted(({c for c in conds if c != 'full'} | set(joints)))
    if 'full' not in conds or not _precheck(recs, required, cfg, 'development',
                                            cfg['reps']['risk_estimate'], args.rehearsal):
        if 'full' not in conds:
            print('缺少完整提示（full）的对照记录')
        return 1
    tasks = {t.task_id: t for t in load_tasks(include_excluded=True)}
    type_of = {tid: task_features(t).task_type for tid, t in tasks.items()}
    table = estimate(recs, _groups(), type_of, b=cfg['stats']['bootstrap'], seed=cfg['stats']['seed'],
                     min_exposed_groups=cfg['prism']['min_exposed_groups'], pairs=pairs)
    untested = {k: v['untested'] for k, v in table.meta['interactions'].items() if 'untested' in v}
    if untested:
        print(f'互补组交互无法检验的 {len(untested)} 项（两组的单独条目不写入，部署时回退完整提示）：')
        for k, why in untested.items():
            print(f'  {k:40s} {why}')
    missing = sorted(k for k, c in table.meta['coverage'].items()
                     if '+' not in k.split('|')[0] and c['exposed_groups'] < table.meta['min_exposed_groups'])
    print(f"覆盖诊断：{len(table.meta['coverage'])} 个条目，暴露题目组少于 {table.meta['min_exposed_groups']} "
          f"的单组条目 {len(missing)} 个（不写入风险表，部署时回退完整提示）")
    for k in missing:
        c = table.meta['coverage'][k]
        print(f"  {k:40s} 暴露 {c['exposed_runs']}/{c['runs']} 次，{c['exposed_tasks']} 题，{c['exposed_groups']} 组")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.save(out)
    print(f'风险表 {len(table.entries)} 项，合并组 {table.merged} → {out}')
    return 0


def _required_joints(conds, pairs) -> list[str]:
    """配置中的互补组在某状态下两组都做了单独消融时，必须有对应的联合消融（任一顺序书写均可）。"""
    out = []
    for a, b in pairs:
        for c in conds:
            if c.startswith(f'ablate:{a}@'):
                phase = c.rsplit('@', 1)[1]
                if f'ablate:{b}@{phase}' in conds and f'ablate:{b}+{a}@{phase}' not in conds:
                    out.append(f'ablate:{a}+{b}@{phase}')
    return out


def _precheck(recs, conditions, cfg, split, reps, rehearsal) -> bool:
    problems = check_records(recs, conditions, cfg, split, reps, rehearsal)
    if rehearsal:
        print(REHEARSAL_BANNER)
    if problems:
        print('输入检查未通过，不给出结论：')
        for p in problems:
            print('  - ' + p)
        return False
    return True


def cmd_calibrate(args, cfg):
    try:
        recs = load_records([Path(p) for p in args.runs])
    except ExperimentMismatch as e:
        print(f'拒绝校准：{e}')
        return 1
    eps = cfg['prism']['epsilons_pp']
    delta = cfg['stats']['delta_pp'] / 100
    ordered = [(e, f'prism@{e:g}') for e in eps]
    if not _precheck(recs, ['full'] + [c for _, c in ordered], cfg, 'construction',
                     cfg['reps']['calibrate'], args.rehearsal):
        return 1
    best, steps = stats.fixed_sequence(recs, ordered, 'full', _groups(), delta,
                                       b=cfg['stats']['bootstrap'], seed=cfg['stats']['seed'])
    for s in steps:
        d = s.diff
        print(f'ε={s.epsilon:g}  成功率差 {100 * d.estimate:+.2f}pp  95% 区间 [{100 * d.low:+.2f}, {100 * d.high:+.2f}]  '
              f'{"通过" if s.passed else "未通过"}  （{d.n_tasks} 题 / {d.n_groups} 组）')
    print(f'ε* = {best:g}' if best is not None else '无候选通过：回退完整提示')
    return 0


def cmd_compare(args, cfg):
    try:
        recs = load_records([Path(p) for p in args.runs])
    except ExperimentMismatch as e:
        print(f'拒绝比较：{e}')
        return 1
    groups = _groups()
    conds = sorted({r['condition'] for r in recs} - {args.control})
    if not _precheck(recs, [args.control] + conds, cfg, SPLITS[args.split], cfg['reps'][args.reps_key], args.rehearsal):
        return 1
    kw = {'b': cfg['stats']['bootstrap'], 'seed': cfg['stats']['seed'], 'weighting': args.weighting}
    rows = []
    base = stats.cluster_bootstrap(stats.per_task_mean(recs, args.control), groups, **kw)
    print(f'{args.control}: 成功率 {base.estimate:.3f} [{base.low:.3f}, {base.high:.3f}]')
    for c in conds:
        d = stats.paired_difference(recs, c, args.control, groups, **kw)
        s = stats.relative_saving(recs, c, args.control, groups, b=kw['b'], seed=kw['seed'])
        ok = stats.noninferior(d, cfg['stats']['delta_pp'] / 100)
        mem = [r.get('memory_tokens', 0) for r in recs if r['condition'] == c and r.get('status') != 'infrastructure_error']
        rows.append({'condition': c, 'diff': d.as_dict(), 'saving': s.as_dict(), 'noninferior': ok,
                     'mean_memory_tokens': sum(mem) / len(mem) if mem else 0.0})
        print(f'{c:28s} 成功率差 {100 * d.estimate:+6.2f}pp [{100 * d.low:+6.2f}, {100 * d.high:+6.2f}]  '
              f'输入 token 节省 {100 * s.estimate:5.1f}% [{100 * s.low:5.1f}, {100 * s.high:5.1f}]  '
              f'{"非劣" if ok else "未证明非劣"}  经验 token/题 {rows[-1]["mean_memory_tokens"]:.1f}')
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    return 0


def _insight_file(path, rehearsal: bool) -> list[Experience]:
    """读取洞见文件。正式构建只接受正式提炼的洞见（meta.rehearsal 明确为 false），演练标记不能被覆盖。"""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    flag = (data.get('meta') or {}).get('rehearsal')
    if not rehearsal and flag is not False:
        raise ValueError(f'洞见文件 {path} 是演练产物或来源不明（meta.rehearsal={flag!r}），正式构建不能使用')
    out = []
    for e in data['insights']:
        for k in ('source_tasks', 'source_groups', 'source_splits'):
            e[k] = tuple(e[k])
        out.append(Experience(**e))
    return out


def _memory_records(args, cfg):
    """读取构建记忆库用的记录：只取 nomem 条件；拒绝测试集；正式模式检查覆盖与重复次数。"""
    recs = load_records([Path(p) for p in args.runs])
    mbuild.check_sources(recs)
    others = sorted({r['condition'] for r in recs} - {'nomem'})
    if others:
        print(f'忽略非 nomem 条件的记录：{others}（记忆库只由不带记忆的运行构建）')
    recs = [r for r in recs if r['condition'] == 'nomem']
    if not args.rehearsal:
        if not _precheck(recs, ['nomem'], cfg, None, cfg['reps']['memory_build'], False):
            return None
        want = {t.task_id for t in _load_split(MEMORY_SPLITS)}
        have = {r['task_id'] for r in recs}
        if have != want:
            print(f'输入检查未通过：nomem 覆盖构建集与开发集 {len(have & want)}/{len(want)} 道题')
            return None
    else:
        print(REHEARSAL_BANNER)
    return recs


def cmd_build_memory(args, cfg):
    try:
        recs = _memory_records(args, cfg)
    except (ExperimentMismatch, ValueError) as e:
        print(f'拒绝构建：{e}')
        return 1
    if recs is None:
        return 1
    tasks = {t.task_id: t for t in load_tasks(include_excluded=True)}
    mcfg = cfg['memory']
    entries = mbuild.extract_examples(recs, tasks, mcfg.get('max_answer_chars', 300)) + \
        mbuild.extract_repairs(recs, tasks)
    try:
        if args.insights:
            entries += _insight_file(args.insights, args.rehearsal)
        store = MemoryStore(entries, meta={
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


def retry_rate_limited(call, retries: int, wait: float, sleep=time.sleep):
    """服务商临时限流（HTTP 429）时等待后重试；llm.py 内部的短重试用完后，在这里再等 retries × wait 秒。
    其他错误直接抛出。"""
    from .llm import LLMError
    for attempt in range(retries + 1):
        try:
            return call()
        except LLMError as e:
            if 'HTTP 429' not in str(e) or attempt == retries:
                raise
            print(f'服务商限流，{wait:g} 秒后重试（第 {attempt + 1} 次）', flush=True)
            sleep(wait)


def cmd_distill_insights(args, cfg):
    if not (args.allow_paid or args.rehearsal):
        print('提炼洞见会调用付费接口。确认已获批准后加 --allow-paid 再运行（或用 --rehearsal 演练）。')
        return 2
    try:
        recs = _memory_records(args, cfg)
    except (ExperimentMismatch, ValueError) as e:
        print(f'拒绝提炼：{e}')
        return 1
    if recs is None:
        return 1
    icfg = cfg['memory']['insights']
    tasks = {t.task_id: t for t in load_tasks(include_excluded=True)}
    batches = mbuild.insight_batches(recs, tasks, icfg['batch_size'])
    print(f'{len(batches)} 批“失败—成功”对')
    price, spent, budget = cfg['price_usd_per_mtok'], [0.0], icfg['budget_usd']
    if args.rehearsal:
        def chat(messages):
            return '- Check the result of each operation before answering.\n- Answer in the required format.'
    else:
        from .llm import OpenAICompatibleLLM
        m = cfg['model']
        # --any-provider：只在提炼洞见时放开服务商固定（同一模型），避免一家服务商临时限流卡住流程；洞见文件记录此选项
        extra = {k: v for k, v in (m.get('extra_body') or {}).items() if k != 'provider'} if args.any_provider \
            else m.get('extra_body')
        llm = OpenAICompatibleLLM(m['model'], m['api_base'], m['api_key_env'], m['temperature'], m['top_p'],
                                  m['max_tokens'], m['timeout'], extra)
        counter = TokenCounter()

        from .llm import LLMError

        def chat(messages):
            if spent[0] >= budget:
                raise RuntimeError(f'达到洞见提炼的费用上限 {budget} 美元')
            try:
                r = retry_rate_limited(lambda: llm.chat(messages), args.rate_limit_retries, args.rate_limit_wait)
            except LLMError as e:
                raise RuntimeError(str(e)) from e
            p_in = r.prompt_tokens if r.prompt_tokens is not None else counter.count_messages(messages)
            p_out = r.completion_tokens if r.completion_tokens is not None else counter.count(r.content)
            spent[0] += (p_in * price['input'] + p_out * price['output']) / 1e6
            return r.content
    try:
        insights = mbuild.distill_insights(batches, chat, icfg['max_rules'])
    except RuntimeError as e:
        print(f'停止：{e}')
        return 3
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'meta': {'model': None if args.rehearsal else cfg['model']['model'],
                                        'rehearsal': bool(args.rehearsal), 'n_batches': len(batches),
                                        'cost_usd_est': spent[0], 'any_provider': bool(getattr(args, 'any_provider', False))},
                               'insights': [asdict(e) for e in insights]}, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    print(f'洞见 {len(insights)} 条，估计费用 {spent[0]:.4f} 美元 → {out}')
    return 0


def cmd_show_memory(args, cfg):
    settings = memory_settings(cfg, args.condition)
    if settings is None:
        print('该条件不使用经验记忆')
        return 1
    retriever = make_retriever(cfg, settings, allow_rehearsal=True)
    task = next(t for t in load_tasks(include_excluded=True) if t.task_id == args.task_id)
    groups = retriever.static(task, task_type(task.instruction))
    counter = TokenCounter()
    print(f'题目 {task.task_id}（组 {task.group_id}）：{task.instruction}')
    for g, uids in groups.items():
        for u in uids:
            e = retriever.store.by_id[u.split(':', 2)[2]]
            print(f'--- {g} {u}（来源题目 {list(e.source_tasks)}，{counter.count(e.text)} tokens）\n{e.text}')
    return 0


def cmd_memory_stats(args, cfg):
    recs = load_records([Path(p) for p in args.runs])
    sent: dict[str, list[int]] = {}
    for r in recs:
        if r.get('status') == 'infrastructure_error':
            continue
        for u in {u for t in r.get('turns', []) for u in t.get('experiences', [])}:
            sent.setdefault(u.split(':', 2)[2], []).append(int(r['success']))
    rows = sorted(({'eid': e, 'sent_runs': len(v), 'success_rate': sum(v) / len(v)} for e, v in sent.items()),
                  key=lambda d: (-d['sent_runs'], d['eid']))
    by_kind: dict[str, list] = {}
    for d in rows:
        by_kind.setdefault(d['eid'].split('-')[0], []).append(d)
    for k, ds in sorted(by_kind.items()):
        n = sum(d['sent_runs'] for d in ds)
        print(f"{k:4s} 被发送过的条目 {len(ds):4d}  涉及运行 {n:6d}  "
              f"加权成功率 {sum(d['success_rate'] * d['sent_runs'] for d in ds) / n:.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    return 0


def cmd_token_audit(args, cfg):
    recs = load_records([Path(p) for p in args.runs])
    n = same = 0
    for r in recs:
        for t in r.get('turns', []):
            api = t['api'].get('prompt_tokens')
            if api is not None:
                n += 1
                same += int(api == t['tokens']['input'])
    print(f'有服务商计数的请求 {n} 次，本地计数完全一致 {same} 次' if n else '记录中没有服务商计数')
    return 0


def cmd_show_runs(args, cfg):
    """离线查看运行记录：每局每轮模型做了什么、数据库返回了什么，以及几项诊断计数。不调用模型、不花钱。"""
    from collections import Counter
    tasks = {t.task_id: t for t in load_tasks(include_excluded=True)}
    recs = [r for r in load_records([Path(p) for p in args.runs]) if r.get('status') != 'infrastructure_error']
    recs.sort(key=lambda r: (r.get('condition', ''), r.get('task_id', 0), r.get('rep', 0)))
    status, diag = Counter(), Counter()
    for r in recs:
        t = tasks.get(r.get('task_id'))
        kind = '查询' if t is not None and t.is_query else '修改'
        status[(kind, r['status'], 'OK' if r['success'] else 'FAIL')] += 1
        ops = [x.get('content') or '' for x in r.get('turns', []) if x.get('action') == 'operation']
        if kind == '查询' and any(re.match(r'\s*(INSERT|UPDATE|DELETE|REPLACE)\b', o, re.I) for o in ops):
            diag['查询题中执行了修改语句'] += 1
        if kind == '查询' and r['status'] == 'answered' and (r.get('answer') or '').strip().lower() in ('done', '[]', '[()]'):
            diag['查询题答案为 done/[]/[()]'] += 1
        if any((x.get('exec') or {}).get('ok') is False for x in r.get('turns', [])):
            diag['出现过 SQL 错误'] += 1
        if args.brief:
            continue
        print(f"===== {r.get('condition')} 题 {r.get('task_id')} 第 {r.get('rep')} 次 [{kind}] {r['status']} "
              f"{'OK' if r['success'] else 'FAIL'}")
        if args.instructions and t is not None:
            print('  题目:', t.instruction[:args.width])
        for x in r.get('turns', []):
            print('  >>', (x.get('content') or x.get('response') or '')[:args.width].replace('\n', ' '))
            if 'observation' in x:
                print('  <<', x['observation'][:args.width].replace('\n', ' '))
    n = len(recs)
    ok = sum(int(r['success']) for r in recs)
    print(f'\n共 {n} 局，成功 {ok}（{ok / n:.1%}）' if n else '没有记录')
    for k, v in sorted(status.items()):
        print(f'  {k[0]} {k[1]:15s} {k[2]:4s} {v}')
    for k, v in diag.items():
        print(f'  {k}: {v}')
    return 0


def cmd_reparse(args, cfg):
    """离线复查：用当前解析规则重新解析记录中每条回复的原始文本，统计动作格式。不调用模型、不花钱。
    对比记录中的解析结果与当前规则的结果；操作是否执行成功不在此列（需重跑）。"""
    from collections import Counter
    from .parser import first_action, parse
    recs = [r for r in load_records([Path(p) for p in args.runs]) if r.get('status') != 'infrastructure_error']
    old, new, label, examples = Counter(), Counter(), Counter(), []
    for r in recs:
        for t in r.get('turns', []):
            raw = t.get('raw_response') or ''
            p_new = parse(first_action(raw))
            old[t.get('action') if t.get('action') != 'invalid' else f"invalid:{t.get('parse_reason')}"] += 1
            new[p_new.kind if p_new.kind != 'invalid' else f'invalid:{p_new.reason}'] += 1
            m = re.search(r'Action:[ \t]*([^\n]*)', raw)
            label[(m.group(1).strip()[:20] if m else None, '```' in raw)] += 1
            if p_new.kind == 'invalid' and len(examples) < args.examples:
                examples.append(raw[:400])
    print(f'有效记录 {len(recs)} 条，回复 {sum(old.values())} 条')
    print('记录中的解析结果：', dict(old.most_common()))
    print('当前规则的解析结果：', dict(new.most_common()))
    print('首个 Action 标签与是否含代码块：', label.most_common(12))
    for e in examples:
        print('--- 仍无效的回复示例 ---')
        print(e)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='prism_thesis')
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'default.json'))
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('check-data')
    p = sub.add_parser('verify-env')
    p.add_argument('--split', default='all', choices=SPLITS)
    p = sub.add_parser('show-prompt')
    p.add_argument('--task-id', type=int, required=True)
    p.add_argument('--condition', default='full')
    for name in ('dry-run', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--split', default='dev', choices=SPLITS)
        p.add_argument('--conditions', default='full')
        p.add_argument('--reps', type=int, default=1)
        p.add_argument('--limit', type=int)
        p.add_argument('--task-ids')
        p.add_argument('--workers', type=int)
        p.add_argument('--out', required=True)
        if name == 'dry-run':
            p.add_argument('--agent', default='gold', choices=['gold', 'gold_after_error'])
            p.add_argument('--resume', action='store_true')
        else:
            p.add_argument('--allow-paid', action='store_true')
            p.add_argument('--budget-usd', type=float)
    p = sub.add_parser('risk-estimate')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--single-only', action='store_true')
    p.add_argument('--rehearsal', action='store_true', help='演练：只做一致性检查，结果不得用于冻结或论文')
    p = sub.add_parser('calibrate')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--rehearsal', action='store_true', help='演练：只做一致性检查，结果不得用于冻结或论文')
    p = sub.add_parser('compare')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--split', default='test', choices=[k for k in SPLITS if k not in ('all', 'memory')])
    p.add_argument('--rehearsal', action='store_true', help='演练：只做一致性检查，结果不得用于冻结或论文')
    p.add_argument('--control', default='full')
    p.add_argument('--weighting', default='task', choices=['task', 'group'])
    p.add_argument('--reps-key', default='compare', help='配置 reps 中预定次数表的键（k 敏感性实验用 k_sensitivity）')
    p.add_argument('--json')
    for name in ('build-memory', 'distill-insights'):
        p = sub.add_parser(name)
        p.add_argument('--runs', nargs='+', required=True, help='构建集与开发集上 nomem 条件的运行记录')
        p.add_argument('--rehearsal', action='store_true', help='演练：不检查覆盖与模型，结果不得用于正式实验')
        if name == 'build-memory':
            p.add_argument('--out', help='默认为配置中的 memory.store')
            p.add_argument('--insights', help='distill-insights 输出的洞见文件')
        else:
            p.add_argument('--out', required=True)
            p.add_argument('--allow-paid', action='store_true')
            p.add_argument('--rate-limit-retries', type=int, default=10)
            p.add_argument('--rate-limit-wait', type=float, default=60)
            p.add_argument('--any-provider', action='store_true',
                           help='提炼洞见时不固定服务商（同一模型由 OpenRouter 自动选择服务商）')
    p = sub.add_parser('show-memory')
    p.add_argument('--task-id', type=int, required=True)
    p.add_argument('--condition', default='full')
    p = sub.add_parser('memory-stats')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--json')
    p = sub.add_parser('show-runs', help='离线查看运行记录的每一轮与诊断计数（不花钱）')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--brief', action='store_true', help='只打印汇总')
    p.add_argument('--instructions', action='store_true', help='同时打印题目')
    p.add_argument('--width', type=int, default=200)
    p = sub.add_parser('reparse', help='离线用当前解析规则重新解析记录中的原始回复（不花钱）')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--examples', type=int, default=5)
    p = sub.add_parser('token-audit')
    p.add_argument('--runs', nargs='+', required=True)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    fn = globals()['cmd_' + args.cmd.replace('-', '_')]
    return fn(args, cfg)


if __name__ == '__main__':
    sys.exit(main())
