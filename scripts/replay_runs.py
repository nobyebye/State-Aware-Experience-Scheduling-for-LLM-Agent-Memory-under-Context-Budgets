"""一致性检查：把已有轨迹中模型的原始回复按本程序的解析规则重放到本程序的 MySQL 环境，重新判分，
与原记录的成功与否对照。不调用模型、不花钱，需要 Docker MySQL。

输入为 JSONL，每行含 task_id、success，以及 turns（每轮的原始回复在 raw 或 raw_response 字段）。
用法：python scripts/replay_runs.py <轨迹文件> [--port 13399]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prism_thesis.data import load_tasks  # noqa: E402
from prism_thesis.env import MySQLServer, TaskSession  # noqa: E402
from prism_thesis.evaluate import match_direct  # noqa: E402
from prism_thesis.parser import first_action, parse  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('runs')
    ap.add_argument('--port', type=int, default=13399)
    ap.add_argument('--examples', type=int, default=5)
    args = ap.parse_args()
    tasks = {t.task_id: t for t in load_tasks(include_excluded=True)}
    rows = [json.loads(line) for line in open(args.runs, encoding='utf-8') if line.strip()]
    table, diffs = collections.Counter(), []
    with MySQLServer(port=args.port) as srv:
        for r in rows:
            t = tasks[int(r['task_id'])]
            answer, status = None, 'round_limit'
            with TaskSession(srv, t) as s:
                for tu in r['turns']:
                    p = parse(first_action(tu.get('raw') or tu.get('raw_response') or ''))
                    if p.kind == 'operation':
                        s.execute(p.content)
                    else:
                        status, answer = ('answered', p.content) if p.kind == 'answer' else ('invalid', None)
                        break
                ok = (s.table_md5() == t.answer_md5 if t.answer_md5 is not None
                      else status == 'answered' and match_direct(answer, t.answer_direct))
            table[('查询' if t.is_query else '修改', bool(r['success']), ok)] += 1
            if bool(r['success']) != ok and len(diffs) < args.examples:
                diffs.append((r['task_id'], bool(r['success']), ok, status, (answer or '')[:120]))
    n = len(rows)
    print(f'{n} 局；原记录成功 {sum(bool(r["success"]) for r in rows) / n:.1%}，'
          f'重放成功 {sum(v for k, v in table.items() if k[2]) / n:.1%}，'
          f'一致 {sum(v for k, v in table.items() if k[1] == k[2]) / n:.1%}')
    for k, v in sorted(table.items()):
        print(f'  {k[0]} 原记录 {"OK" if k[1] else "FAIL"} → 重放 {"OK" if k[2] else "FAIL"}: {v}')
    for d in diffs:
        print('  不一致示例:', d)
    return 0


if __name__ == '__main__':
    sys.exit(main())
