"""把系统指令拆成单元，在固定轨迹下估算 PRISM 手工策略的每题输入 token（零调用）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import collections, re, statistics as st
ctx = common.context(__doc__)
c, sp, FUNCS, SYSTEM, first_action, rows = ctx.c, ctx.sp, ctx.FUNCS, ctx.SYSTEM, ctx.first_action, ctx.rows
L=[l for l in sp.split('\n')]
print([ (i,c(l),l[:50]) for i,l in enumerate(L) if l.strip()])
FL='\nAvailable SQL functions: ' + ', '.join(sorted(FUNCS)) + '.\n'
KW={'AVG':r'\baverage|\bmean\b','COUNT':r'\bcount|how many|number of','SUM':r'\btotal|\bsum\b','MAX':r'\bmax|highest|most|largest|greatest|top|latest|maximum','MIN':r'\bmin|lowest|least|smallest|earliest|minimum|fewest'}
def build(lines_keep, funcs):
    s='\n'.join(L[i] for i in range(len(L)) if i in lines_keep)+'\n'
    if funcs: s+='\nAvailable SQL functions: '+', '.join(sorted(funcs))+'.\n'
    return s
ALL=set(range(len(L)))
OPS={10,11,12}; ANSFMT={13,14}
base=new=0; rec=[0,0]; per=[]
selq=re.compile(r"^(what|which|how many|find|list|show|return|retrieve|select|calculate|count|get|give|identify|determine)\b",re.I)
for r in rows:
    t=r['task']; instr=t['instruction']; schema='\nSchema: '+str(t['columns'])+'\nTable: '+t['table']
    is_q=bool(selq.search(instr.strip()))
    pred={f for f,p in KW.items() if re.search(p,instr,re.I)}
    used=set()
    for tu in r['turns']:
        for e in ((tu['step'] or {}).get('sql_events') or []):
            for f in FUNCS:
                if re.search(r'\b'+f+r'\s*\(', e.get('sql') or '', re.I): used.add(f)
    rec[0]+=len(used&pred); rec[1]+=len(used)
    hist=0; msgs=2; preverr=False; b=n=0
    for i,tu in enumerate(r['turns']):
        b+=c(SYSTEM)+c(instr+schema)+hist+5*msgs+5
        keep=set(ALL)
        if not is_q: keep-=ANSFMT
        if i==0 or preverr:
            sysm=build(keep, FUNCS if preverr else pred); task=instr+schema
        else:
            sysm=build(keep-OPS, set()); task=instr
        n+=c(sysm)+c(task)+hist+5*msgs+5
        a=c(first_action(tu['raw'] or '')); hist+=a; msgs+=1
        stp=tu['step'] or {}
        if stp.get('observation') is not None and not stp.get('done'):
            hist+=c(stp['observation']); msgs+=1
            preverr='MySQL error' in stp['observation'] or 'error' in stp['observation'][:40].lower()
    base+=b; new+=n
print('function recall %d/%d'%tuple(rec))
print('base %.1f new %.1f reduction %.1f%%'%(base/len(rows),new/len(rows),100*(1-new/base)))
print('query-intent tasks', sum(bool(selq.search(r['task']['instruction'].strip())) for r in rows))
