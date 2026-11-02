# Zero-call analysis of runs/development_20261004_curve_v1. 2026-10-17
# 用法: python analysis/first_turn_errors_dev_curve.py --runs-dir <旧仓库>/runs/development_20261004_curve_v1（只读）
import argparse,json,os,collections
ap=argparse.ArgumentParser(); ap.add_argument('--runs-dir',required=True); root=ap.parse_args().runs_dir
if not os.path.isdir(root): raise SystemExit('找不到运行目录: '+root)
out=collections.defaultdict(collections.Counter); sample=None
for d in os.listdir(root):
    p=os.path.join(root,d,'events.jsonl')
    if not os.path.exists(p): continue
    ev=[json.loads(l) for l in open(p,encoding='utf-8')]
    s=[e for e in ev if e['event']=='run_started'][0]['spec']
    fin=[e for e in ev if e['event']=='run_finished']
    if not fin or fin[0]['result'].get('success') is None: continue
    cond=f"{s['method']}@{s['budget']}/{s.get('memory_format','full')}/u{s.get('min_utility',0)}/r{s.get('redundancy',0)}"
    steps=[e for e in ev if e['event']=='benchmark_step']
    if not steps: continue
    o=(steps[0]['result'].get('observation') or '')
    ses=steps[0]['result'].get('sql_events') or []
    err=any(x.get('error') for x in ses) or 'error' in o.lower()[:200]
    out[cond]['err0' if err else 'ok0']+=1
    succ=fin[0]['result']['success']
    out[cond][('err0' if err else 'ok0')+('_S' if succ else '_F')]+=1
    if sample is None and len(steps)>1: sample=(steps[0]['result'],)
for c,v in sorted(out.items()): print(c,dict(v))
print(json.dumps(sample)[:800])
