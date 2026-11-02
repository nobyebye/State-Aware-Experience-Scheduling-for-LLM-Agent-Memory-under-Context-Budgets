# 零调用功效估算：用既有 330 次基础运行的每题成功率 p_i 作为生成随机性的来源，
# 模拟“压缩条件 vs 完整提示”的配对比较，估计成功率差 95% 区间半宽与非劣检验功效。
import json, collections, numpy as np, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
by=collections.defaultdict(list); grp={}
for r in rows: by[r['task_id']].append(int(r['success'])); grp[r['task_id']]=r['group_id']
p={t:np.mean(v) for t,v in by.items()}
groups=collections.defaultdict(list)
for t in p: groups[grp[t]].append(p[t])
G=list(groups.values())
print('tasks',len(p),'groups',len(G),'base success %.3f'%np.mean(list(p.values())))
print('task-level p distribution: 0:%d  1/3:%d  2/3:%d  1:%d'%tuple(sum(abs(x-v)<1e-9 for x in p.values()) for v in (0,1/3,2/3,1)))
rng=np.random.default_rng(0)
def sim(nG,R,f,delta_true,S=4000,delta=0.05):
    ests=[];passes=0
    for _ in range(S):
        idx=rng.integers(0,len(G),nG); ps=np.concatenate([G[i] for i in idx])
        q=ps.copy()
        # 异质效应：比例 f 的题目成功率下降 0.5，比例 f+delta_true*? 保持均值差约为 delta_true
        u=rng.random(len(ps))
        down=u<f; up=(u>=f)&(u<2*f)
        q=np.where(down,np.clip(ps-0.5,0,1),q); q=np.where(up,np.clip(ps+0.5,0,1),q)
        q=np.clip(q-delta_true,0,1)
        a=rng.binomial(R,ps)/R; b=rng.binomial(R,q)/R
        # 题目组层面的配对差
        d=b-a; sizes=[len(G[i]) for i in idx]; gd=np.add.reduceat(d,np.cumsum([0]+sizes[:-1]))/np.array(sizes)
        est=gd.mean(); se=gd.std(ddof=1)/np.sqrt(nG)
        ests.append(est); passes+= (est-1.96*se) > -delta
    return 1.96*np.std(ests), passes/S
print('\n组数 G, 每臂重复 R, 异质比例 f, 真实下降 → 95%区间半宽(百分点), δ=5pp 非劣通过率')
for nG in (22,33,67,100,243):
  for R in (1,3,10):
    for f,dt in ((0,0),(0.1,0),(0.1,0.02)):
      hw,pw=sim(nG,R,f,dt,S=1500)
      print(f'G={nG:3d} R={R:2d} f={f:.1f} Δ={dt:.2f}  半宽={hw*100:5.1f}  通过率={pw:.2f}')
