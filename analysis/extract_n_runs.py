"""从旧仓库开发曲线运行日志中抽取无经验基础运行（方法 N），供第一步与第四步分析使用（零调用）。

用法:
    python analysis/extract_n_runs.py --runs-dir <旧仓库>/runs/development_20261004_curve_v1
默认输出 analysis/n_runs_extract.jsonl；输出路径必须位于本项目内，旧仓库只读。
已有输出时须加 --overwrite；全部读完、校验通过后才写入（先写临时文件再替换），失败不会截断已有输出。
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent


def extract(root):
    out, diag = [], collections.Counter()
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, 'events.jsonl')
        if not os.path.exists(p):
            continue
        with open(p, encoding='utf-8') as f:
            ev = [json.loads(l) for l in f]
        start = [e for e in ev if e['event'] == 'run_started'][0]['spec']
        fin = [e for e in ev if e['event'] == 'run_finished']
        if start['method'] != 'N' or not fin or fin[0]['result'].get('success') is None:
            continue
        task = [e for e in ev if e['event'] == 'selection'][0]['task']
        turns = []
        resp = [e for e in ev if e['event'] == 'model_response']
        steps = [e['result'] for e in ev if e['event'] == 'benchmark_step']
        trims = {e['turn']: e for e in ev if e['event'] == 'action_trimmed'}
        # 诊断（不改变配对方式）：按位置配对的前提是每轮恰有一条响应与一条环境步骤
        if len(resp) != len(steps):
            diag['响应数与步骤数不等的运行'] += 1
        for i, r in enumerate(resp):
            try:
                content = r['response']['choices'][0]['message']['content']
                usage = r['response'].get('usage', {})
            except Exception:
                content, usage = None, {}
            turns.append({'raw': content, 'usage': usage, 'trimmed': i in trims,
                          'step': steps[i] if i < len(steps) else None})
        out.append({'run': d, 'task_id': start['task_id'], 'group_id': start['group_id'],
                    'task': task, 'success': fin[0]['result']['success'],
                    'status': fin[0]['result']['status'], 'turns': turns})
    return out, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs-dir', type=Path, required=True, help='旧仓库 runs/development_20261004_curve_v1（只读）')
    ap.add_argument('--out', type=Path, default=ANALYSIS_DIR / 'n_runs_extract.jsonl')
    ap.add_argument('--overwrite', action='store_true', help='允许替换已有输出')
    a = ap.parse_args()
    runs, out = a.runs_dir.resolve(), a.out.resolve()
    if not runs.is_dir():
        sys.exit(f'找不到运行目录: {runs}')
    if PROJECT_ROOT not in out.parents:
        sys.exit(f'输出必须位于本项目 {PROJECT_ROOT} 内: {out}')
    if out.exists() and not a.overwrite:
        sys.exit(f'{out} 已存在；确认要替换时加 --overwrite')
    rows, diag = extract(runs)
    if not rows:
        sys.exit('没有找到符合条件的运行，未写入')
    tmp = out.with_name(out.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, out)
    print(len(rows))
    for k, v in diag.items():
        print(f'# {k}: {v}', file=sys.stderr)


if __name__ == '__main__':
    main()
