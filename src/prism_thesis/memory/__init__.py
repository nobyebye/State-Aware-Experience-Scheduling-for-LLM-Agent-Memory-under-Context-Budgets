"""经验记忆库：结构化经验条目的存储（store）、词法索引（bm25）、错误签名（signature）、
状态感知检索（retrieval）与离线构建（build）。

经验条目只由构建集与开发集上的运行记录整理得到，测试集题目不进入记忆库；检索时排除与当前题目同组的条目。
在线检索只用 BM25 与规则匹配，不调用模型；洞见由模型离线提炼（只在构建记忆库时调用）。
"""
from .store import EXP_GROUPS, Experience, MemoryStore, is_exp, render_memory
from .retrieval import Retriever

__all__ = ['EXP_GROUPS', 'Experience', 'MemoryStore', 'Retriever', 'is_exp', 'render_memory']
