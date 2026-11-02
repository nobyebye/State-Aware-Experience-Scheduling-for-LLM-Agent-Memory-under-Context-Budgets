# Zero-call analysis of runs/development_20261004_curve_v1. 2026-10-17
# 用法: python analysis/token_breakdown_dev_curve.py --runs-dir <旧仓库>/runs/development_20261004_curve_v1（只读）
import argparse,json,os,glob,collections,statistics as st
ap=argparse.ArgumentParser(); ap.add_argument('--runs-dir',required=True); root=ap.parse_args().runs_dir
if not os.path.isdir(root): raise SystemExit('找不到运行目录: '+root)
agg=collections.defaultdict(lambda: collections.defaultdict(list))
for d in os.listdir(root):
    p=os.path.join(root,d,'events.jsonl')
    if not os.path.exists(p): continue
    ev=[json.loads(l) for l in open(p,encoding='utf-8')]
    st0=[e for e in ev if e['event']=='run_started']
    fin=[e for e in ev if e['event']=='run_finished']
    if not st0 or not fin: continue
    s=st0[0]['spec']; r=fin[0]['result']
    if r.get('success') is None: continue
    cond=f"{s['method']}@{s['budget']}/{s.get('memory_format','full')}/u{s.get('min_utility',0)}/r{s.get('redundancy',0)}"
    ctx=[e for e in ev if e['event']=='context']
    a=agg[cond]
    a['turns'].append(len(ctx))
    a['mem'].append(sum(c['memory_tokens'] for c in ctx))
    a['nonmem'].append(sum(c['nonmemory_tokens'] for c in ctx))
    a['nonmem_t0'].append(ctx[0]['nonmemory_tokens'] if ctx else 0)
    a['succ'].append(1 if r['success'] else 0)
    a['succ_t1'].append(1 if (r['success'] and len(ctx)<=1) else 0)
    tw=collections.Counter(len(ctx) for _ in [0])
for c,a in sorted(agg.items()):
    n=len(a['turns'])
    tc=collections.Counter(a['turns'])
    print(c,n,'succ%.3f'%st.mean(a['succ']),'turns%.2f'%st.mean(a['turns']),dict(sorted(tc.items())),
      'mem%.0f'%st.mean(a['mem']),'nonmem%.0f'%st.mean(a['nonmem']),'nonmem_t0 %.0f'%st.mean(a['nonmem_t0']))
