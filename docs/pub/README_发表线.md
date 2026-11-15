# 发表线代码说明（阶段 0）

本分支（`claude/project-thread-ro8zkd`）从硕士论文的代码分支 `claude/project-thread-bj67bn` 分出，只用于投稿论文。
硕士论文的命令行（`python -m prism_thesis`）、db_bench 环境与全部已有测试不变。

## 新增内容

| 内容 | 位置 |
| --- | --- |
| 环境接口（提示单元、解析、执行、判分、恢复映射、人工规则、经验文本） | `src/prism_thesis/envs/base.py` |
| db_bench 环境（委托原模块，行为不变） | `src/prism_thesis/envs/dbbench.py` |
| BIRD-Agent 环境（只读 SQLite，多轮，官方 EX 判分） | `src/prism_thesis/envs/bird.py` |
| 操作系统环境（LifelongAgentBench os_interaction，Docker） | `src/prism_thesis/envs/os_shell.py`、`docker/os/Dockerfile` |
| 数据准备 | `scripts/pub/prepare_bird.py`、`scripts/pub/prepare_os.py` |
| 测试集封存检查与使用记录 | `src/prism_thesis/pub/seal.py` |
| 发表线命令行 | `python -m prism_thesis.pub --config configs/pub/<bird|os>.json <命令>` |
| 测试 | `tests/test_pub_envs.py`（容器测试需 `PRISM_TEST_DOCKER=1`） |

可在新环境直接使用的条件：`full`、`nomem`、`phase_switch`、`random_drop`、`ablate:<组>@<状态>`、`js-ref`、
`js-abl:<组>+<组>@<状态>`、`js@λ`。依赖风险表的 PRISM / static 与 LLMLingua-2 基线目前只在 db_bench 实现，
属于阶段 1 的工作。

## 与执行方案的一处不同：BIRD 用原生 SQLite

执行方案原写“把 BIRD 转成 MySQL 任务”。实现时改为原生 SQLite：BIRD 的参考 SQL 是 SQLite 方言，官方执行准确率也在
SQLite 上计算；MySQL 版（Mini-Dev）只有 500 题，跨方言比较结果不可靠。用 SQLite 可以使用全部 dev 题目并保持官方判分。
交互协议：每轮一条 SQL，确认后 `Final Answer: submit` 提交最近一次成功查询的结果（与 InterCode 相同）。

## 在 VM 上准备数据（不调用模型，不花钱）

```bash
cd ~/prism-thesis && git fetch origin claude/project-thread-ro8zkd && git checkout claude/project-thread-ro8zkd
source .venv/bin/activate && export PYTHONPATH=src

# 1. BIRD dev：从官方页面 https://bird-bench.github.io/ 下载 dev.zip 并解压
#    （解压后有 dev.json 与 dev_databases.zip，后者再解压一次）
python scripts/pub/prepare_bird.py --dev-json <路径>/dev.json --db-root <路径>/dev_databases
mkdir -p data/bird && ln -s <路径>/dev_databases data/bird/dev_databases     # 数据库文件不进仓库
python -m prism_thesis.pub --config configs/pub/bird.json check-data --hash-dbs
python -m prism_thesis.pub --config configs/pub/bird.json verify-env --split all
python -m prism_thesis.pub --config configs/pub/bird.json show-prompt --task-id <题号> --condition nomem

# 2. 操作系统环境：从 HuggingFace csyq/LifelongAgentBench 下载 os_interaction/train-00000-of-00001.parquet
python scripts/pub/prepare_os.py --raw <路径>/train-00000-of-00001.parquet      # 会构建镜像并在容器中逐题核对
python -m prism_thesis.pub --config configs/pub/os.json check-data
python -m prism_thesis.pub --config configs/pub/os.json verify-env --split all

# 3. 提交生成的数据文件（data/bird/、data/os/），之后划分不再改动
```

`prepare_bird.py` 的输出最后一行是汇总（题数、各划分大小、排除原因）。划分目标为 build 350 / search 250 /
calib 400 / test 500；可用题目不足时按比例缩小，此时需要回到执行方案重新核对检验功效。

## 尚未核对的事项

- os_interaction 原始文件的字段格式是按官方代码推断的（本次无法在线读取数据行）；`prepare_os.py` 同时接受嵌套对象
  与字面量字符串，若字段名不同会在转换时报错，届时按实际字段调整。
- BIRD 完整提示的 token 数要等真实数据生成后用 `show-prompt` 测量，再据此重算预算。
- OpenRouter 上各模型的价格与可用服务商尚未核对（付费运行前完成）。

## 测试集封存

`run --split test` 必须加 `--prereg <已提交的预先登记文件>`；代码、配置、数据有未提交改动时拒绝运行；同一环境与模型
的测试集第二次运行必须给出 `--rerun-reason`。每次放行都记入 `docs/pub/test_usage.jsonl`，该文件随代码提交，论文中
如实报告。测试集不能用 `--limit` 取子集，测试集记录不能进入记忆库。
