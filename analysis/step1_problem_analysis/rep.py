"""跨轮重复发送占比与失败类别。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import collections
L = common.context(__doc__)
c, SYSTEM, first_action, rows = L.c, L.SYSTEM, L.first_action, L.rows
tot=rep=0; succ_turn=collections.defaultdict(list)
fail=collections.Counter()
for r in rows:
    t=r['task']; user=t['instruction']+'\nSchema: '+str(t['columns'])+'\nTable: '+t['table']
    S=c(SYSTEM);Q=c(user);hist=0;msgs=2;prev=0
    for i,turn in enumerate(r['turns']):
        inp=S+Q+hist+5*msgs+5; tot+=inp
        if i>0: rep+=prev  # everything sent last turn is re-sent
        new=c(first_action(turn['raw'] or '')); hist+=new; msgs+=1
        stp=turn['step'] or {}
        if stp.get('observation') is not None and not stp.get('done'): hist+=c(stp['observation']); msgs+=1
        prev=inp
    if not r['success']:
        st_=r['status']; obs=' '.join((tu['step'] or {}).get('observation') or '' for tu in r['turns'])
        fail[st_ if st_!='completed' else ('sql_error_seen' if 'MySQL error' in obs else 'wrong_answer')]+=1
print('repeated share of input %.1f%%'%(100*rep/tot)); print(fail)
