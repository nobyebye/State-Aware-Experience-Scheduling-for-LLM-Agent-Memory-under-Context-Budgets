"""输出 token 浪费、任务描述长度、按操作类型的成功率、系统提示逐行 token。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import collections, statistics as st
L = common.context(__doc__)
c, SYSTEM, first_action, rows = L.c, L.SYSTEM, L.first_action, L.rows
raw=kept=0; per=[]
schema=[];instr=[]; ops=collections.Counter(); succ_by_op=collections.defaultdict(list)
for r in rows:
    t=r['task']; schema.append(c('\nSchema: '+str(t['columns'])+'\nTable: '+t['table'])); instr.append(c(t['instruction']))
    for turn in r['turns']:
        out=turn['usage'].get('completion_tokens') or 0; k=c(first_action(turn['raw'] or ''))
        raw+=out; kept+=min(k,out)
    sqls=[e.get('sql','') for tu in r['turns'] if tu['step'] for e in (tu['step'].get('sql_events') or [])]
    op=(sqls[0].split()[0].upper() if sqls and sqls[0] else 'NONE')
    ops[op]+=1; succ_by_op[op].append(r['success'])
print('output tokens raw %.1f kept %.1f wasted %.1f%%'%(raw/len(rows),kept/len(rows),100*(1-kept/raw)))
print('instruction %.1f schema+table %.1f'%(st.mean(instr),st.mean(schema)))
print({k:(v,round(st.mean(succ_by_op[k]),2)) for k,v in ops.items()})
# system prompt line tokens
for line in SYSTEM.split('\n'):
    if line.strip(): print(c(line), line[:90])
