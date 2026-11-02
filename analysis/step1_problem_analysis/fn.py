"""SQL 函数列表的实际使用情况。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import collections, statistics as st, re
L = common.context(__doc__)
FUNCS, rows = L.FUNCS, L.rows
cnt=[];allf=collections.Counter()
for r in rows:
    used=set()
    for tu in r['turns']:
        for e in ((tu['step'] or {}).get('sql_events') or []):
            for f in FUNCS:
                if re.search(r'\b'+f+r'\s*\(', e.get('sql') or '', re.I): used.add(f)
    cnt.append(len(used)); allf.update(used)
print('funcs listed',len(FUNCS),'mean used %.2f'%st.mean(cnt),collections.Counter(cnt)); print(allf.most_common(12))
