"""生成操作系统环境数据：python scripts/pub/prepare_os.py --raw <os_interaction 的 parquet 或 entry_dict.json>

原始文件：HuggingFace 数据集 csyq/LifelongAgentBench 的 os_interaction/train-00000-of-00001.parquet。
需要 Docker（在容器里核对每道题的初始化、参考解法与评测）。不调用模型，不产生费用。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from prism_thesis.config import load_config  # noqa: E402
from prism_thesis.pub.os_prep import DEFAULT_SIZES, prepare  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'pub' / 'os.json'))
    ap.add_argument('--seed', type=int, default=20261115)
    ap.add_argument('--skip-verify', action='store_true', help='不在容器中核对（只用于调试，不能用于正式数据）')
    for name, n in DEFAULT_SIZES.items():
        ap.add_argument(f'--{name}', type=int, default=n)
    a = ap.parse_args()
    prepare(Path(a.raw), load_config(a.config), {k: getattr(a, k) for k in DEFAULT_SIZES}, a.seed, a.skip_verify)
