"""任务环境。每个环境把“题目、提示单元、动作解析、执行、判分、经验文本”打包成一个 Environment 对象，
智能体循环、策略、检索与运行器都只通过它访问环境相关的内容。

  db_bench  LifelongAgentBench 数据库任务（MySQL）：硕士论文使用的环境，行为与原实现逐字节相同；
  bird      BIRD-Agent：BIRD dev 的问题改为多轮只读 SQLite 智能体任务（发表线主环境）；
  os        LifelongAgentBench 操作系统任务（Docker 中的 Ubuntu shell；发表线第二环境）。

配置文件用 "env" 选择环境，缺省为 db_bench。
"""
from __future__ import annotations

from .base import Environment

_DEFAULT = None


def default_env() -> Environment:
    global _DEFAULT
    if _DEFAULT is None:
        from .dbbench import DbBenchEnv
        _DEFAULT = DbBenchEnv()
    return _DEFAULT


def get_env(cfg: dict) -> Environment:
    name = cfg.get('env', 'db_bench')
    if name == 'db_bench':
        return default_env()
    if name == 'bird':
        from .bird import BirdEnv
        return BirdEnv(cfg)
    if name == 'os':
        from .os_shell import OsEnv
        return OsEnv(cfg)
    raise ValueError(f'未知环境：{name}')
