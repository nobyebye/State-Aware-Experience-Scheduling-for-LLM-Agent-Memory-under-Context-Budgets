"""零调用功效与区间宽度模拟（第二版，2026-10-22 修订）。

用法（在项目根目录或任意目录）:
    python analysis/power_analysis/power_v2.py
    python analysis/power_analysis/power_v2.py --data <n_runs_extract.jsonl> --sims 2000 --bootstrap-check 300
数据: 既有 330 次基础运行（开发集 110 题、67 组，每题 3 次），默认读 analysis/n_runs_extract.jsonl。

目标总体: 开发集题目的等权分布（每道题权重 1/110），与正式分析“按题平均”的估计对象一致。
抽样模型（保留组内相关，同时使每个题目位的边际分布等于目标总体）:
  - 目标集的每个大小为 k 的组：按组大小成比例（概率 k_g / 110）抽一个开发集题目组，
    再在组内等概率有放回抽 k 道题。每个题目位抽到题 i 的概率 = (k_g/110)·(1/k_g) = 1/110。
    （修订前为等概率抽组，题 i 的边际概率为 1/(67·k_g)，与校准时的等权分布不一致。）
  - 每题基线成功概率 p_i = (成功次数 + 0.5) / (3 + 1)，避免 0/1 端点。
  - 压缩条件 logit(q_i) = logit(p_i) + b + sigma·z，z ~ N(0,1)，每个题目位独立抽取。
  - b 的校准: 用 Gauss–Hermite 求积计算 E_z[q_i]，二分法使 mean_i(E_z[q_i] − p_i) = Delta，
    以 80 点与 160 点求积结果之差作为数值误差，要求 < 1e-9。
  - 两臂各运行 R 次，成功次数独立二项抽样。
估计量: 按题平均的成功率差。
推断: 以题目组为单位的线性化（比率估计）标准误，双侧 95% 区间 = 估计 ± 1.96·SE；
      非劣通过 = 区间下界 > −delta（delta = 5 个百分点）。Delta = −5pp 时的通过率即边界处的误判率。
自检（任一不满足即停止，不输出该情形的功效结论）:
  1. 抽样分布期望差: 对每次模拟记录所抽题目位的 mean(E_z[q] − p)，其平均与 Delta 之差
     不超过 4 倍蒙特卡洛标准误与 0.05 个百分点中的较大者；
  2. 估计量平均与 Delta 之差不超过 4 倍蒙特卡洛标准误。
--bootstrap-check N: 对部分情形另做 N 次模拟，比较线性化区间与正式分析所用的按组 bootstrap
      百分位区间（与 src/prism_thesis/stats.py 相同的按题加权重抽样）。
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / 'n_runs_extract.jsonl'
DELTA_NI = 0.05
COMPOSITION = {  # 组大小: 组数（排除已批准的题目后）
    'dev': {1: 47, 2: 10, 3: 4, 4: 3, 5: 2, 9: 1},
    'test': {1: 91, 2: 6, 3: 2, 5: 1},
    'construction': {1: 224, 2: 12, 3: 3, 4: 1, 5: 1, 6: 1},
}


def logit(x):
    return np.log(x / (1 - x))


def expit(x):
    return 1 / (1 + np.exp(-x))


def load(path):
    succ, grp = collections.defaultdict(list), {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            succ[r['task_id']].append(int(r['success']))
            grp[r['task_id']] = r['group_id']
    tasks = sorted(succ)
    p = np.array([(sum(succ[t]) + 0.5) / (len(succ[t]) + 1) for t in tasks])
    gid = {}
    for i, t in enumerate(tasks):
        gid.setdefault(grp[t], []).append(i)
    groups = [np.array(v) for v in gid.values()]
    raw = float(np.mean([np.mean(succ[t]) for t in tasks]))
    return p, groups, raw


def expected_q(p, b, sigma, nodes=80):
    """E_z[expit(logit(p) + b + sigma·z)]，z ~ N(0,1)，概率论 Hermite 求积。"""
    if sigma == 0:
        return expit(logit(p) + b)
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    return expit(logit(p)[:, None] + b + sigma * x[None]) @ w


def solve_b(p, sigma, target, tol=1e-9):
    lo, hi = -10.0, 10.0
    for _ in range(100):
        b = (lo + hi) / 2
        if (expected_q(p, b, sigma) - p).mean() < target:
            lo = b
        else:
            hi = b
    b = (lo + hi) / 2
    err_root = abs((expected_q(p, b, sigma) - p).mean() - target)
    err_quad = abs(expected_q(p, b, sigma, 80).mean() - expected_q(p, b, sigma, 160).mean())
    if err_root > tol or err_quad > tol:
        sys.exit(f'校准失败: sigma={sigma} Delta={target} 求根误差={err_root:.2e} 求积误差={err_quad:.2e}')
    return b


def size_list(name, rng):
    if name == 'dev_third':
        full = np.repeat(list(COMPOSITION['dev']), list(COMPOSITION['dev'].values()))
        return rng.choice(full, 22, replace=False)
    c = COMPOSITION[name]
    return np.repeat(list(c), list(c.values()))


def draw(rng, name, p, groups, gprob):
    sizes = size_list(name, rng)
    picks = rng.choice(len(groups), size=len(sizes), p=gprob)
    ti = np.concatenate([rng.choice(groups[g], k, replace=True) for g, k in zip(picks, sizes)])
    return sizes, ti


def sim(rng, name, R, sigma, target, p, groups, gprob, S, boot_b=0):
    b = solve_b(p, sigma, target)
    eq = expected_q(p, b, sigma)
    hws, ests, oracle, passes = [], [], [], 0
    boot_pass = boot_hw = 0.0
    for _ in range(S):
        sizes, ti = draw(rng, name, p, groups, gprob)
        G, n = len(sizes), sizes.sum()
        q = expit(logit(p[ti]) + b + sigma * rng.standard_normal(len(ti)))
        d = rng.binomial(R, q) / R - rng.binomial(R, p[ti]) / R
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        Sg = np.add.reduceat(d, starts)
        est = Sg.sum() / n
        se = np.sqrt(G / (G - 1) * ((Sg - est * sizes) ** 2).sum()) / n
        hws.append(1.96 * se); ests.append(est); oracle.append((eq[ti] - p[ti]).mean())
        passes += (est - 1.96 * se) > -DELTA_NI
        if boot_b:
            counts = rng.multinomial(G, np.full(G, 1.0 / G), size=boot_b).astype(float)
            bs = (counts @ Sg) / (counts @ sizes)
            lo, hi = np.quantile(bs, [0.025, 0.975])
            boot_pass += lo > -DELTA_NI; boot_hw += (hi - lo) / 2
    ests, oracle = np.array(ests), np.array(oracle)
    tol_o = max(4 * oracle.std(ddof=1) / np.sqrt(S), 0.0005)
    tol_e = 4 * ests.std(ddof=1) / np.sqrt(S)
    ok = abs(oracle.mean() - target) <= tol_o and abs(ests.mean() - target) <= tol_e
    out = dict(oracle=oracle.mean(), est=ests.mean(), hw=float(np.mean(hws)), pass_rate=passes / S, ok=ok,
               tol_o=tol_o, tol_e=tol_e)
    if boot_b:
        out.update(boot_pass=boot_pass / S, boot_hw=boot_hw / S)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--sims', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261022)
    ap.add_argument('--bootstrap-check', type=int, default=0, metavar='N',
                    help='另做 N 次模拟比较线性化区间与按组 bootstrap（0 表示不做）')
    a = ap.parse_args()
    p, groups, raw = load(a.data)
    N = len(p)
    gprob = np.array([len(g) for g in groups], dtype=float) / N
    sha = hashlib.sha256(a.data.read_bytes()).hexdigest()
    print(f'data={a.data.name} sha256={sha[:16]} numpy={np.__version__}')
    print(f'tasks={N} groups={len(groups)} base_raw={raw:.3f} base_smoothed={p.mean():.4f} '
          f'seed={a.seed} S={a.sims} delta={DELTA_NI}')
    print('抽样: 按组大小成比例抽组、组内等概率抽题（每个题目位的边际分布为题目等权）')
    print('数据集 R sigma Delta | 期望差(pp) 估计平均(pp) 半宽(pp) 通过率 自检')
    rng = np.random.default_rng(a.seed)
    failed = []
    for name in ('dev_third', 'dev', 'test', 'construction'):
        for R in (3, 10):
            for sigma in (0.0, 0.5, 1.0):
                for target in (0.0, -0.02, -0.05):
                    r = sim(rng, name, R, sigma, target, p, groups, gprob, a.sims)
                    flag = '通过' if r['ok'] else '未通过'
                    if not r['ok']:
                        failed.append((name, R, sigma, target))
                    pr = f"{r['pass_rate']:.3f}" if r['ok'] else '  —  '
                    print(f"{name:12s} {R:2d}  {sigma:.1f}  {target:+.2f} | {r['oracle']*100:+6.2f}  "
                          f"{r['est']*100:+6.2f}  {r['hw']*100:5.1f}  {pr}  {flag}")
    if a.bootstrap_check:
        print(f'\n线性化区间 与 按组 bootstrap 百分位区间（B=2000）对比，每种情形 {a.bootstrap_check} 次模拟')
        print('数据集 R sigma Delta | 线性化: 半宽 通过率 | bootstrap: 半宽 通过率')
        rng2 = np.random.default_rng(a.seed + 1)
        for name, R in (('test', 10), ('construction', 10), ('dev_third', 10)):
            for sigma in (0.0, 1.0):
                for target in (0.0, -0.05):
                    r = sim(rng2, name, R, sigma, target, p, groups, gprob, a.bootstrap_check, boot_b=2000)
                    print(f"{name:12s} {R:2d}  {sigma:.1f}  {target:+.2f} | {r['hw']*100:5.1f}  {r['pass_rate']:.3f} | "
                          f"{r['boot_hw']*100:5.1f}  {r['boot_pass']:.3f}")
    if failed:
        print('\n以下情形自检未通过，未给出通过率:', failed)
        sys.exit(1)
    print('\n全部情形自检通过。')


if __name__ == '__main__':
    main()
