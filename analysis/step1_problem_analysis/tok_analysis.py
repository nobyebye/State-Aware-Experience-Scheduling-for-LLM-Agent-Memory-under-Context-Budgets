"""重建每轮输入并计数，与服务商 prompt_tokens 核对，给出输入构成与各轮长度。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import collections, statistics as st
L = common.context(__doc__)
c, SYSTEM, first_action, rows = L.c, L.SYSTEM, L.first_action, L.rows
agg=collections.defaultdict(list); per_turn=collections.defaultdict(lambda: collections.defaultdict(list))
prov_match=0; prov_total=0; obs_tok=[]; obs_big=0
for r in rows:
    t=r['task']; user=t['instruction']+'\nSchema: '+str(t['columns'])+'\nTable: '+t['table']
    S=c(SYSTEM); Q=c(user); hist=0; tot=dict(system=0,task=0,history_actions=0,history_obs=0,overhead=0,output=0)
    msgs=2
    for i,turn in enumerate(r['turns']):
        k=i
        per_turn[k]['input'].append(S+Q+hist+5*msgs+5)
        tot['system']+=S; tot['task']+=Q; tot['overhead']+=5*msgs+5
        tot['history_actions']+=sum(x for x in [r.setdefault('_a',[])] and r['_a'])
        tot['history_obs']+=sum(r.setdefault('_o',[]))
        pt=turn['usage'].get('prompt_tokens')
        if pt is not None:
            prov_total+=1; prov_match+= (pt==S+Q+hist+5*msgs+5)
        tot['output']+=turn['usage'].get('completion_tokens',0) or 0
        content=turn['raw'] or ''
        kept=first_action(content)
        a=c(kept); r['_a'].append(a); hist+=a; msgs+=1
        stp=turn['step'] or {}
        if stp.get('observation') is not None and not stp.get('done'):
            o=c(stp['observation']); r['_o'].append(o); obs_tok.append(o); hist+=o; msgs+=1
            if o>200: obs_big+=1
    T=len(r['turns'])
    for k,v in tot.items(): agg[k].append(v)
    agg['turns'].append(T); agg['succ'].append(r['success'])
inp=[sum(agg[k][i] for k in ['system','task','history_actions','history_obs','overhead']) for i in range(len(rows))]
print('runs',len(rows),'success %.3f'%st.mean(agg['succ']),'mean turns %.2f'%st.mean(agg['turns']))
print('provider prompt_tokens exact match %d/%d'%(prov_match,prov_total))
tot_in=st.mean(inp)
for k in ['system','task','history_actions','history_obs','overhead']:
    print('%-16s %.1f  %.1f%%'%(k,st.mean(agg[k]),100*st.mean(agg[k])/tot_in))
print('input total %.1f, output total %.1f'%(tot_in, st.mean(agg['output'])))
for k in sorted(per_turn): print('turn',k,'n',len(per_turn[k]['input']),'mean input %.0f'%st.mean(per_turn[k]['input']))
print('obs tokens mean %.1f median %s p90 %s max %s, >200: %d'%(st.mean(obs_tok),st.median(obs_tok),sorted(obs_tok)[int(.9*len(obs_tok))],max(obs_tok),obs_big))
# outcome categories
print(collections.Counter(r['status'] for r in rows).most_common())
print('turns dist',collections.Counter(agg['turns']))
# diff check
diffs=collections.defaultdict(list)
for r in rows:
    t=r['task']; user=t['instruction']+'\nSchema: '+str(t['columns'])+'\nTable: '+t['table']
    hist=0; msgs=2
    for i,turn in enumerate(r['turns']):
        pt=turn['usage'].get('prompt_tokens')
        mine=c(SYSTEM)+c(user)+hist+5*msgs+5
        if pt: diffs[i].append(pt-mine)
        hist+=r['_a'][i]; msgs+=1
        stp=turn['step'] or {}
        if stp.get('observation') is not None and not stp.get('done'): hist+=c(stp['observation']); msgs+=1
for k,v in diffs.items(): print(k, collections.Counter(v).most_common(6))
