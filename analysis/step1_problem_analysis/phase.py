"""各轮动作类型（操作 / 作答 / 无效）与动作序列分布。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import collections
L = common.context(__doc__)
first_action, rows = L.first_action, L.rows
kind=collections.Counter(); seq=collections.Counter(); invalid=collections.Counter()
for r in rows:
    ks=[]
    for tu in r['turns']:
        a=first_action(tu['raw'] or '')
        k='Op' if a.startswith('Action: Operation') else ('Ans' if a.startswith('Action: Answer') else 'Invalid')
        ks.append(k)
    seq[' > '.join(ks)]+=1
    for i,k in enumerate(ks): kind[(i,k)]+=1
print(sorted(kind.items())); print(seq.most_common(8))
