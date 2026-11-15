"""生成 BIRD-Agent 数据：python scripts/pub/prepare_bird.py --dev-json <dev.json> --db-root <dev_databases>

BIRD dev 可从官方网站 https://bird-bench.github.io/ 下载（dev.zip，解压后含 dev.json 与 dev_databases/）。
输出写到 data/bird/（题目、划分、表信息、数据库文件清单）。不调用模型，不产生费用。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from prism_thesis.pub.bird_prep import DEFAULT_SIZES, prepare  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev-json', required=True)
    ap.add_argument('--db-root', required=True)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'bird'))
    ap.add_argument('--seed', type=int, default=20261115)
    ap.add_argument('--timeout', type=float, default=30)
    for name, n in DEFAULT_SIZES.items():
        ap.add_argument(f'--{name}', type=int, default=n)
    a = ap.parse_args()
    prepare(Path(a.dev_json), Path(a.db_root), Path(a.out), {k: getattr(a, k) for k in DEFAULT_SIZES},
            a.seed, a.timeout)
